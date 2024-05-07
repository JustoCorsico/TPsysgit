import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.io import wavfile
sd.Stream(samplerate=44100, blocksize=None, device=None, channels=None,
dtype=None, latency=None, extra_settings=None, callback=None,
finished_callback=None, clip_off=None, dither_off=None,
never_drop_input=None, prime_output_buffers_using_stream_callback=None)