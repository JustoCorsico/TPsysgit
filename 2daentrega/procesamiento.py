import numpy as np
from tkinter import *
from IPython.display import clear_output, display
from tkinter import filedialog
import soundfile as sf
from scipy import signal
from funciones import cargar_wav
from funciones import esc_log
from scipy.io.wavfile import write
from funciones import time_plot
from funciones import reproducir
from funciones import analisis_frecuencias
import matplotlib.pyplot as plt


## Función de carga de archivos de audio (dataset)
files_list = []
wav_list = []
def select_files():
    '''
    Carga archivos de audio en formato '.wav', '.wma', '.mp3'
    y los almacena en una lista de tuplas (Numpy array, frecuencia de muestreo).
        
    '''    
    clear_output()
    files_list.clear() 
    wav_list.clear() 
    root = Tk()
    root.withdraw() # Hide the main window.
    root.call('wm', 'attributes', '.', '-topmost', True) # Raise the root to the top of all windows.
    files_list.append(filedialog.askopenfilenames(filetypes = [('Wav', '.wav'),('Mp3', '.mp3'),('Wma', '.wma')])) # List of selected files will be set button's file attribute.  
    print(files_list)
    
    for i in range(len(files_list[0])):  # Almacena los datos de los archivos en una lista.
        wav = cargar_wav(files_list[0][i])
        wav_list.append(wav)
      
    return(files_list, wav_list)

## Función de sintetización de respuesta al impulso
def IR_sint(f_i, RT_xfrec_list, tiempo_impulso=6, frec_muestreo=44100):
    '''
    Genera una respuesta al impulso sintetizada a partir de los parámetros acústicos brindados.
       
    '''    
    t = np.arange(0, tiempo_impulso*frec_muestreo)/frec_muestreo
    y_i_list = []
    for i in range(len(f_i)):
        tau_i = np.log(10**(-3))/RT_xfrec_list[i]
        y_i = (np.exp(tau_i*t))*np.cos(2*np.pi*f_i[i]*t)
        y_i_list.append(y_i)

    y = sum(y_i_list)
    # Normalizado
    y_max = max(abs(max(y)), abs(min(y)))
    y_norm = y/y_max 
    # Guardar archivo wav
    write('IR_sint.wav', frec_muestreo, y_norm)
    # Info del numpy array
    #print(type(y_norm), y_norm)
    return(y_norm)

## Función obtener respuesta al impulso
def respuesta_impulso(rec_sine_sweep, invfilter, nombre_impulso="impulso"):
    """
    Función que genera un impulso a través de la convolución un sinesweep logarítmico grabado y un filtro inverso.

    """

    data_sweep, fs = sf.read(rec_sine_sweep)  
    data_invfilter, fs = sf.read(invfilter)

    impulso = signal.fftconvolve(data_sweep, data_invfilter, mode='full')
    impulso_max = max(abs(max(impulso)), abs(min(impulso)))
    impulso_norm = impulso/impulso_max 
    sf.write("{}.wav".format(nombre_impulso), impulso_norm, fs)

    return impulso_norm

## Función conversión a escala logarítmica
def esc_log(data_impulse, a = 20):
    """
    Convierte una señal a escala logarítmica normalizada.

    """
    A = data_impulse/(np.max(np.abs(data_impulse)))  #normalizacion 
    np.seterr(divide = 'ignore', invalid = 'ignore' ) # daba errores de division por cero y valor no valido
    Norm_log = a * np.log10(A)
    return Norm_log

## Función filtros norma IEC 61260
def filtro_IEC(archivo):
    """
    Filtra una señal en bandas de octava.

    """
    audiodata, fs = sf.read(archivo)
    lista_filtros = []
    nominal_frec = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for i in range(len(nominal_frec)):
        #Octava - G = 1.0/2.0 / 1/3 de Octava - G=1.0/6.0
        G = 1.0/2.0
        factor = np.power(2, G)
        centerFrequency_Hz = nominal_frec[i]

        #Calculo los extremos de la banda a partir de la frecuencia central
        lowerCutoffFrequency_Hz = centerFrequency_Hz/factor;
        upperCutoffFrequency_Hz = centerFrequency_Hz*factor;
        if upperCutoffFrequency_Hz >= (fs/2):
            upperCutoffFrequency_Hz = (fs/2)-1
            
        
        # Extraemos los coeficientes del filtro 
        b,a = signal.iirfilter(4, [2*np.pi*lowerCutoffFrequency_Hz,2*np.pi*upperCutoffFrequency_Hz],
                                    rs=60, btype='band', analog=True,
                                    ftype='butter') 

        # para aplicar el filtro es más óptimo
        sos = signal.iirfilter(4, [lowerCutoffFrequency_Hz,upperCutoffFrequency_Hz],
                                    rs=60, btype='band', analog=False,
                                    ftype='butter', fs=fs, output='sos') 
        w, h = signal.freqs(b,a)

        # aplicando filtro al audio
        filt = signal.sosfilt(sos, audiodata)
        
        list = [centerFrequency_Hz, filt, h, w]
        lista_filtros.append(list)
        #sf.write("Frecuencia {}.wav".format(lista_filtros[i][0]), lista_filtros[i][1], fs)
        print('Frecuencia de corte inferior: ', round(lowerCutoffFrequency_Hz), 'Hz')
        print('Frecuencia central: ', centerFrequency_Hz, 'Hz')
        print('Frecuencia de corte superior: ', round(upperCutoffFrequency_Hz), 'Hz')
        
    return(lista_filtros)

if __name__ == "__main__":
    files_names, wav_files = select_files()
    
    # Bandas de Octava según IEC 61260
    nominal_frec = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    # Genero un vector con T60i para cada frecuencia con los datos de Openair.com, Usina del Arte Symphony Hall 
    RT_xfrec_list = [2.15, 1.48, 1.63, 1.91, 2.08, 2.09, 1.82, 1.6, 1.18, 1.11]

    ir_test = IR_sint(nominal_frec, RT_xfrec_list)
    time_plot(ir_test, 44100, 'IR Prueba')
    reproducir('IR_sint.wav')

    rec_sine_sweep = "sine_sweepDrive.wav" # descargado de Gdrive
    invfilter = "filtro_inversoDrive.wav" # descargado de Gdrive
    respuesta_impulso(rec_sine_sweep, invfilter)
    data, fs = sf.read("impulso.wav")
    nombre_grafico = "Impulso test"
    time_plot(data, fs, nombre_grafico)
    
    file = 'C:/Users/noUser/Desktop/tps/TPsysgit/2daentrega/Mono.wav' # descargado de Gdrive
    frec = filtro_IEC(file) # funcion filtros norma IEC61260
    print(frec[6][0])
    analisis_frecuencias(frec[6][1], fs) 

    data, fs = cargar_wav(file)
    data_log = esc_log(data, 20) # funcion escala logaritmica normalizada
    time_plot(data_log, fs, nombre_grafico="impulso_log")
    
    