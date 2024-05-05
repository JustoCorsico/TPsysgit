
import numpy as np
from matplotlib import pyplot as plt
import simpleaudio as sa
from scipy.io import wavfile
import pandas as pd
import soundfile as sf

# FUNCIONES
# Sine sweep

def get_data():
    f1 = int(input('Ingrese frecuencia inferior:'))
    f2 = int(input('Ingrese frecuencia superior:'))
    T = int(input('Ingrese T: '))
    return (f1, f2, T)

def get_signal(f1, f2, T):
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w1=2*np.pi*f1
    w2=2*np.pi*f2   
    R = np.log(w2/w1)
    K = float((T*w1)/R) 
    L = T/R
    fs=44100
    t = np.linspace(0,T,T*fs)
    f = np.sin(K*(np.exp(t/L)-1))
    return (f, t, R, K, L, w1, fs)

def norm_signal(f,kt):
    g = f / np.max(np.abs(f))
    j = kt / np.max(np.abs(kt))
    return (g,j)

def get_signalwav(n,fs,j):
    audio1 = (n * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("Sine Sweep.wav",fs, audio1)
    audio2 = (j * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write("Filtro inverso.wav",fs, audio2)
# Filtro Inverso

def get_fi(f, K, L, T, t, w1):
    ssi = np.flip(f)
    wt = (K/L)*np.exp(t/L)
    mt = w1/(2*np.pi*wt)
    kt = mt*ssi
    return (kt, ssi)

def get_convolve(f, kt):
    conv = np.convolve(f, kt)
    return(conv)

def get_plot(t, f, kt, ssi, conv):
    plt.plot(t,f)
    plt.xlim((0,2))
    plt.show()
    plt.plot(t, ssi)
    plt.xlim((0,2))
    plt.show()
    plt.plot(t, kt)
    plt.xlim((0,2))
    plt.show()
    plt.plot(conv)
    plt.show()
    
    return ()

# MAIN


f1, f2, T = get_data()

f, t, R, K, L, w1, fs = get_signal(f1, f2, T)

ssi, kt = get_fi(f, K, L, T, t, w1)

n,j = norm_signal(f,kt)

get_signalwav(n,fs,kt)

conv = get_convolve(f, kt)

get_plot(t, f, ssi, kt, conv)

















