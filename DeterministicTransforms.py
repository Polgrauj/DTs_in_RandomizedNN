import numpy as np
import math 


# Discrete Hartley Transform
def DHT(x):
    x = np.fft.fft(x, axis=0)
    x = x.real - x.imag
    return x


# Fast Walsh Hadamard Transform
def FWH_transform(x, N, sequency): 
    # Sequency = 0: coefficients in hadamard order
    # Sequency = 1: coefficients in sequency-walsh order
    m = int(math.log2(N))
    y = np.zeros(x.shape)
    for i in range(0, m): # for log2 N stages
        n = int(pow(2, m-1-i)) # length of section
        k = 0
        while k < N-1: # for all sections in a stage
            for j in range(0, n): # for all points in a section
                j1 = k + j
                t = x[j1] + x[j1+n]
                x[j1+n] = x[j1] - x[j1+n]
                x[j1] = t
            k = k + (2*n) # // move on to next section
    w = 1 / np.sqrt(N)
    for i in range(0, N):
        x[i] = x[i] * w
    if sequency == 1:  # converting to sequency (Walsh) order
        for i in range(0, N):
            j = hadamard2walsh(i, m)
            y[i] = x[j]
        for i in range(0, N):
            x[i] = y[i]
    return x
    
def hadamard2walsh(i, m): # Converts a sequency index i to Hadamard index j
    i = i^(i>>1); # Gray code
    j = 0
    for k in range(0, m):
        j = (j << 1) | (1 & (i >> k)) # bit reversal
    return j


# Slant Transform
def Slant_transform(x, N):
    if N==2: # 2-point transform
        u = x[0]
        v = x[1]
        x[0] = (u+v) / np.sqrt(2)
        x[1] = (u-v) / np.sqrt(2)    
    else:
        y1 = np.zeros((x.shape))
        y2 = np.zeros((x.shape))
        for n in range(0, int(N/2)):
            y1[n] = x[n] + x[int(N/2)+n]
            y2[n] = x[n] - x[int(N/2)+n]
        Slant_transform(y1, int(N/2)) # recursion
        Slant_transform(y2, int(N/2))
        for n in range(0, int(N/2)):
            x[n] = y1[n] / np.sqrt(2)
            x[int(N/2)+n] = y2[n] / np.sqrt(2)
        w = 4*N*N-4
        c = np.sqrt(3*N*N / w)
        s = np.sqrt((N*N-4) / w)
        u= x[int(N/4)] 
        v = x[int(N/2)]
        x[int(N/4)] = c*u-s*v # rotation
        x[int(N/2)] = s*u + c*v
    return x