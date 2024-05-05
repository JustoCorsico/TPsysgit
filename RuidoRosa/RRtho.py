import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import soundfile as sf
#Informacion necesaria
w1 = float(input("Frecuencia angular inferior = "))
w2 = float(input("Frecuencia angular superior = "))
T = float(input("Tiempo T : "))
fs = 44100
duracion = 10
#sine sweep
def sine_sweep(w1, w2, T, fs):
  R = np.log(w2 / w1)
  L = T / R
  K = (T * w1) / R
  t = np.linspace(0, T, int(T * fs))
  return np.sin(K * (np.exp(t / L) - 1))
#Normalización
def normalize1(señal):
  return señal / np.max(np.abs(señal))
#Ploteo
def plot_signal1(señal, Sine_Sweep):
  plt.figure()
  plt.plot(señal)
  plt.title(Sine_Sweep)
  plt.xlabel('Tiempo')
  plt.ylabel('Amplitud')
  plt.grid(True)
  plt.show()
#Audio
def generate1_wav(señal, Sine_Sweep, fs):
  audio1 = (señal * np.iinfo(np.int16).max).astype(np.int16)
  wavfile.write(Sine_Sweep, fs, audio1)
#Filtro inverso
def filtro_inverso(w1, w2, T, fs):
     R = np.log(w2 / w1)
     L = T / R
     K = (T * w1) / R
     h = np.linspace(0,T, int(fs*T))
     t = np.linspace(0, T, int(T * fs))
     return  (w1 / (K/L * np.exp(h/L) * np.pi * 2)) * np.flip(sine_sweep(w1, w2, T, fs))
#Normalización
def normalize2(señalInv):
  return señalInv / np.max(np.abs(señalInv))
#Audio
def generate2_wav(señalInv, Filtro_Inverso, fs):
  audio2 = (señalInv * np.iinfo(np.int16).max).astype(np.int16)
  wavfile.write(Filtro_Inverso, fs, audio2)
#Ploteo
def plot_signal2(señalInv, Filtro_Inverso):
  plt.figure()
  plt.plot(señalInv)
  plt.title(Filtro_Inverso)
  plt.xlabel('Tiempo')
  plt.ylabel('Amplitud')
  plt.grid(True)
  plt.show()
#Ruido rosa
def voss(duracion, fs, ncols=16):
    nrows = int(duracion * fs)
    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    # el numero total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)
    df = pd.DataFrame(array)
    filled = df.ffill(axis=0)
    total = filled.sum(axis=1)
    ## Centrado de el array en 0
    total = total - total.mean()
    # Asegurar que los datos estén en el rango [-1, 1]
    total = total / np.max(np.abs(total))
    # Guardar el archivo de audio .wav
    sf.write('ruidoRosa.wav', total, fs, format='WAV', subtype='PCM_16')
    return total
#Ploteo
def plot_signal3(total, Ruido_Rosa):
  plt.figure()
  plt.plot(total)
  plt.title(Ruido_Rosa)
  plt.xlabel('Tiempo')
  plt.ylabel('Amplitud')
  plt.grid(True)
  plt.show()
#Llamamos a las funciones
señal = normalize1(sine_sweep(w1, w2, T, fs))
señalInv = normalize2(filtro_inverso(w1, w2, T, fs))
ruido_rosa = voss(duracion, fs)
#plot_signal1(señal, "Sine Sweep")
#plot_signal2(señalInv, "Filtro Inverso")
#plot_signal3(ruido_rosa, "Ruido Rosa" )
#generate1_wav(señal, "Sine_Sweep.wav", fs)
#generate2_wav(señalInv, "Filtro_Inverso.wav", fs)
#Prueba si estamos bien
"""
import scipy.signal as sg
convolucion = sg.convolve(señal, señalInv)
plt.plot(convolucion)
plt.show()
"""