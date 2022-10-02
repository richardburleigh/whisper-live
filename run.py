import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import whisper
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
from datetime import datetime
import time
import os
import mutagen
from mutagen.wave import WAVE


import ray

ray.init()

@ray.remote(num_gpus=1)
class MyModel():

    def __init__(self):
        # load the model once in the constructor
        self.model = whisper.load_model("large")
        print("Loaded model on device: ", self.model.device)

    def run(self, i):
        # ...
        # code to perform prediction using self.predictor_fn
        # ...
        translate_options = {"task" : "translate"}
        translate_options["language"] = "Korean"
        #translate_options["fp16"] = False
        start_time = time.time()
        try:
          self.text = self.model.transcribe(f"tmp{i}.wav", **translate_options)["text"]
        except Exception as e:
          print(e)
        elapsed_time = time.time() - start_time
        source = WAVE(f"tmp{i}.wav")
        #print("Source duration: ", source.info.length)
        #print("Processing time: ", elapsed_time)
        os.remove(f"tmp{i}.wav")
        print(self.text)
        return self.text

    def get(self):
        return self.text


@ray.remote
def getValues(running):
   while True:
        ready_ids, _remaining_ids = ray.wait(running,timeout=0)
        if len(running) > 0:
            if running[0] in ready_ids:
                print(ray.get(running[0]))
                running.remove(running[0])


logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        info = self.pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (self.pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.pa.get_device_info_by_host_api_device_index(0, i).get('name'))

        input_id = input("Which input device? ")
        #self.pa.input_device_index=int(input_id)

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'input_device_index': int(input_id),
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        #logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def main(ARGS):
    # Load model
    #print("Loading model..")
    #model = whisper.load_model("tiny",device="cpu")

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=None)
                         
    translate_options = {"task" : "translate"}
    translate_options["language"] = ARGS.language.lower()
    print("Language is set to ", ARGS.language)
    print(translate_options)

    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()
    text = ""
    for x in range(0, ARGS.threads):
      globals()[f"remoteModel{x}"] = MyModel.remote()
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    wav_data = bytearray()
    result = None
    running = []
    ready_ids = []
    displayed = []
    n = 0
    p = 0




    for frame in frames:
    
        ready_ids, _remaining_ids = ray.wait(running,timeout=0)
        #print(ray.wait(running,timeout=0))
        if len(running) > 0:
            if running[0] in ready_ids:
                print(ray.get(running[0]))
                running.remove(running[0])
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            vad_audio.write_wav(f"tmp{str(n)}.wav", wav_data)
            wav_data = bytearray()
            #print("Start: ",datetime.now())

            #text = model.transcribe("tmp.wav", **translate_options)["text"]
            running.append(globals()[f"remoteModel{p}"].run.remote(n))
            
            p = p + 1
            if p > ARGS.threads -1:
              p = 0


            n = n + 1
            #print("End: ",datetime.now())
            #print(result)
            #print("Recognized: %s" % result)

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-l', '--language', default='Korean',
                        help="Destination language.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help=f"Number of whisper threads.")

    ARGS = parser.parse_args()
    main(ARGS)
