

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import soundfile as sf
import time


def ruidoRosa_voss(nrows, ncols):
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
    
    ## Normalizado
    valor_max = max(abs(max(total)),abs(min(total)))
    total = total / valor_max
    
    # Agregar generación de archivo de audio .wav
    
    return total
def get_data():
    t=int(input("Ingrese tiempo de duracion del ruido rosa:"))
    fs=int(input("Ingrese la frecuencia de muestreo"))
    return(t,fs)

def ruido_rosa_user(t,fs,total):
    sf.write("Ruido rosa.wav",total,fs,format="WAV",subtype="PCM_16")
    return
def get_plot(t,fs,total):
    m=np.linspace(0,t,fs*t)
    plt.plot(m,total)
    plt.show()
    return
#main
t,fs=get_data()
inicio = time.time()


total=ruidoRosa_voss(t*fs,16)
ruido_rosa_user(t,fs,total)
get_plot(t,fs,total)


fin = time.time()
print("La latencia de la funcion es :",fin-inicio)

