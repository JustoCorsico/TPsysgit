# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import soundfile as sf
import sounddevice as sd
from tkinter import *
from IPython.display import clear_output, display
from tkinter import filedialog

def cargar_wav(file):
    
    data, fs = sf.read(file)
    return(data, fs)

def time_plot(data, fs, nombre_grafico=" "):
    """
    Genera el gráfico del dominio temporal de una señal.

    """
    rate = len(data)       
    time = np.linspace(0, rate/fs, num=rate)  # Objeto Numpy para la duración en el eje x
        
    # Grafico
    plt.figure(figsize=(15, 5))
    plt.plot(time, data, linewidth=0.5)
    plt.title(f'Gráfico {nombre_grafico} Dominio del tiempo')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo [s]')
    plt.show()
    return()

def reproducir(filename):
    'Función para reproducir audio'

    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    status = sd.wait()  # Wait until file is done playing
    return data

def esc_log(data_impulse, a = 20):
    """
    Convierte un array a escala logarítmica. 

    """
    A = data_impulse/(np.max(np.abs(data_impulse)))
    Norm_log = a * np.log10(A)
    return Norm_log

def analisis_frecuencias(audio, fs):
    '''
    Calcular la transformada de Fourier de la señal de audio y graficarla.

    '''        
    audio = audio / np.max(np.abs(audio))  # Normalizar los valores de la señal
    fft_data = np.fft.fft(audio)

    # Calcular los valores de frecuencia correspondientes
    fft_freq = np.fft.fftfreq(len(audio), 1.0 / fs)

    # Tomar solo la mitad de los datos (la otra mitad es simétrica)
    fft_data = esc_log(np.abs(fft_data[:len(audio)//2]))
    fft_freq = fft_freq[:len(audio)//2]

    # Graficar el análisis de frecuencia
    plt.figure(figsize=(10, 5))
    plt.semilogx(fft_freq, fft_data)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.title('Análisis de frecuencia de audio')
    plt.grid(True)
    custom_xticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks(custom_xticks, custom_xticks)
    plt.fill_between(fft_freq, fft_data, np.min(fft_data))
    plt.show()
    return(fft_data)

   
