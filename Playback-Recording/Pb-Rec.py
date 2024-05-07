import sounddevice as sd
import numpy as np
import soundfile as sf 
from scipy.io import wavfile
import time
from matplotlib import pyplot as plt

def get_data():
    t=int(input("Ingrese tiempo de grabacion y reproducción:"))
    print(sd.query_devices())
    mic=int(input("Ingrese numero de dispositivo de entrada:"))
    spkr=int(input("Ingrese numero de dispositivo de salida:"))
    return(mic,spkr,t)
##Seteo Grabación
def playrecord(mic,spkr,t):
    sd.default.samplerate = frec
    sd.default.device=mic,spkr
    sd.default.channels=2
    myrecording = sd.playrec(data,channels=1)
    time.sleep(t)
    sd.stop() 
    return myrecording
##Generar wav
def get_wav(myrecording):
    audio2 = (myrecording* np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("grabacion.wav",frec,audio2)
    return
##main
data, frec = sf.read("Sine Sweep.wav", dtype='float32') ##Toma la data del archivo
mic,spkr,t=get_data()
myrecording=playrecord(mic,spkr,t)
get_wav(myrecording)