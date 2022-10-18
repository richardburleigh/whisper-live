import auditok
import ray
import os
import time
from scipy.io import wavfile
import pathlib
import ctypes
from halo import Halo

language = "ko"
task = "translate"
duration = 30 # Record 30 seconds of audio before processing with Whisper
modelsize = "large"
showQueue = False

libname = "libwhisper.so"

if not os.path.exists(libname):
    print("Unable to find ", libname, ". Please build it as per https://github.com/ggerganov/whisper.cpp/issues/9#issuecomment-1272555209")
    exit()

ray.init()

cwd = os.getcwd()
tmpdir = os.path.join(cwd, "temp")
try:
    os.mkdir(tmpdir)
except:
    print("Temp directory already exists.")

@ray.remote
class Whisper():

    def __init__(self):

        class WhisperFullParams(ctypes.Structure):
            _fields_ = [
                ("strategy",             ctypes.c_int),
                ("n_threads",            ctypes.c_int),
                ("offset_ms",            ctypes.c_int),
                ("translate",            ctypes.c_bool),
                ("no_context",           ctypes.c_bool),
                ("print_special_tokens", ctypes.c_bool),
                ("print_progress",       ctypes.c_bool),
                ("print_realtime",       ctypes.c_bool),
                ("print_timestamps",     ctypes.c_bool),
                ("language",             ctypes.c_char_p),
                ("greedy",               ctypes.c_int * 1),
            ]
        # load the model once in the constructor
        self.model_name = modelsize
        self.tmpdir = tmpdir

        self.libname = pathlib.Path().absolute() / libname
        self.whisper = ctypes.CDLL(self.libname)
        # tell Python what are the return types of the functions
        self.whisper.whisper_init.restype                  = ctypes.c_void_p
        self.whisper.whisper_full_default_params.restype   = WhisperFullParams
        self.whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p
         # initialize whisper.cpp context
        self.ctx = self.whisper.whisper_init(("models/ggml-" + self.model_name + ".bin").encode("utf-8"))
        # get default whisper parameters and adjust as needed
        self.params = self.whisper.whisper_full_default_params(0)
        self.params.print_realtime = True
        self.params.print_progress = False
        self.params.language = language.encode()
        if str(task) == "translate":
            self.params.translate = True
        else:
            self.params.translate = False
        self.params.n_threads = os.cpu_count() - 1
        self.is_running = True
        print("Running on ", self.params.n_threads, " threads.")

    def run(self, i, startTime):
        start_time = time.time()

        self.txt = ""

        try:
            wavpath = os.path.join(self.tmpdir,f"tmp{i}.wav")
 
            #print(wavpath)
            wavpath = str.encode(wavpath)

            # load WAV file
            samplerate, data = wavfile.read(wavpath)

            # convert to 32-bit float
            data = data.astype('float32')/32768.0

            # run the inference
            result = self.whisper.whisper_full(ctypes.c_void_p(self.ctx), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))
            if result != 0:
                print("Error: {}".format(result))
                exit(1)

            txtfull = ""

            n_segments = self.whisper.whisper_full_n_segments(ctypes.c_void_p(self.ctx))
            for i in range(n_segments):
                t0  = self.whisper.whisper_full_get_segment_t0(ctypes.c_void_p(self.ctx), i)
                t1  = self.whisper.whisper_full_get_segment_t1(ctypes.c_void_p(self.ctx), i)
                txt = self.whisper.whisper_full_get_segment_text(ctypes.c_void_p(self.ctx), i)
                txtfull = txtfull + "\n" + txt.decode('utf-8').strip()

            self.txt = txtfull
        except Exception as e:
            print(e)
            return ""
        return "[" + str(startTime)+ "] " + self.txt.strip()

sr = 16000
sw = 2
ch = 1
eth = 55 # alias for energy_threshold, default value is 50

running = []
eady_ids = []
displayed = []
n = 0

remoteModel = Whisper.remote()

spinner = Halo(spinner='line')
data = None
spinner.start()
try:
    for region in auditok.split(input=None, sr=sr, sw=sw, ch=ch, eth=eth):
        ready_ids, _remaining_ids = ray.wait(running,timeout=0)
        if len(running) > 0:
            if running[0] in ready_ids:
                spinner.stop()
                print(ray.get(running[0]))
                spinner.start()
                running.remove(running[0])

        if data == None:
            data = region
        else:
            data = data + region
        if data.duration > duration:
            data.save(os.path.join(tmpdir,f"tmp{n}.wav"))
            running.append(remoteModel.run.remote(n, time.strftime("%X")))
            if showQueue:
               print("Queue: ", len(_remaining_ids), " Ready: ", len(ready_ids))
            data = None
            n = n + 1
except KeyboardInterrupt:
     pass
