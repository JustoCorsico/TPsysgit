import numpy as np
from matplotlib import pyplot as plt

# Sine sweep

w1 = float(input('Ingrese frecuencia inferior:'))
w2 = float(input('Ingrese frecuencia inferior:'))
T = int(input('Ingrese T: '))
R = np.log(w2/w1)
K = float((T*w1)/R)
L = T/R
fs=10000
t = np.linspace(0,T,T*fs)

f = np.sin(K*(np.exp(t/L)-1))

plt.plot(t,f)
plt.show()

# Filtro Inverso

wt = (K/L)*np.exp(t/L)
mt = w1/(2*np.pi*wt)
R1 = np.log(w1/w2)
L1 = T/R1
f1 = f = np.sin(K*(np.exp(t/L1)-1))

kt = mt*f1

plt.plot(t,kt)
plt.show()

conv = np.convolve(f,kt)

plt.plot(t,conv)
plt.show