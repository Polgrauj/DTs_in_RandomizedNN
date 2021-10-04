import numpy as np
from MyFunctions import *
from scipy.fftpack import fft, dct, dst
from scipy.linalg import hadamard
from sympy import fwht
import pywt 
from DeterministicTransforms import *
import math
import time


def DT_options(Yi, Yi_test, DT_it, Q, test_true):
    if DT_it == 2 or DT_it == 3 or DT_it == 12:
        zero_pad = 2**(math.ceil(math.log2(Yi.shape[0]))) - Yi.shape[0] 
        Yi = np.concatenate((Yi, np.zeros((zero_pad, Yi.shape[1]))), axis=0)
        if test_true: 
            Yi_test = np.concatenate((Yi_test, np.zeros((zero_pad, Yi_test.shape[1]))), axis=0)
    else:
        zero_pad = 0
 # Chosing the deterministic transform, does not need to work with testing part            
    N = Yi.shape[0]
    dec_lev = math.floor(math.log2(N))
    Zi_part2_test = []
    if DT_it == 0: # Discrete Cosine Transform
        Zi_part2_train = dct(Yi , type=2, axis=0, norm='ortho', overwrite_x=False)
        del Yi
        if test_true:
            Zi_part2_test = dct(Yi_test, type=2, axis=0, norm='ortho', overwrite_x=False)
    elif DT_it == 1: # Discrete Sine Transform
        Zi_part2_train = dst(Yi , type=1, axis=0, norm='ortho', overwrite_x=False)
        del Yi
        if test_true:
            Zi_part2_test = dst(Yi_test, type=1, axis=0, norm='ortho', overwrite_x=False)
    elif DT_it == 2: # Fast Walsh Hadamard Transform: Coefficients in normal Hadamard order
        Zi_part2_train = FWH_transform(Yi, N, 0)
        del Yi
        if test_true:
            Zi_part2_test = FWH_transform(Yi_test, N, 0)
    elif DT_it == 3: # Fast Walsh Hadamard Transform: Coefficients in order of increasing sequency value, where each row has an additional zero crossing
        Zi_part2_train = FWH_transform(Yi, N, 1)
        del Yi
        if test_true:
            Zi_part2_test = FWH_transform(Yi_test, N, 1)
    elif DT_it == 4: # Discrete Hartley Transform
        Zi_part2_train = DHT(Yi)
        del Yi
        if test_true:
            Zi_part2_test = DHT(Yi_test)
    elif DT_it == 5: # Haar Transform
        coeffs_train = pywt.wavedec(Yi, 'haar', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'haar', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
    elif DT_it == 6:
        coeffs_train = pywt.wavedec(Yi, 'sym2', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'sym2', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    elif DT_it == 7:
        coeffs_train = pywt.wavedec(Yi, 'coif1', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi 
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'coif1', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    elif DT_it == 8:
        coeffs_train = pywt.wavedec(Yi, 'bior1.3', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'bior1.3', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    elif DT_it == 9:
        coeffs_train = pywt.wavedec(Yi, 'rbio1.1', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi 
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'rbio1.1', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    elif DT_it == 10: # Daubeuchies 4 Transform
        coeffs_train = pywt.wavedec(Yi, 'db4', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'db4', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    elif DT_it == 11: # Daubeuchies 
        coeffs_train = pywt.wavedec(Yi, 'db20', level=dec_lev, axis=0)
        Zi_part2_train = np.concatenate(coeffs_train)
        del coeffs_train, Yi
        if test_true:
            coeffs_test = pywt.wavedec(Yi_test, 'db20', level=dec_lev, axis=0)
            Zi_part2_test = np.concatenate(coeffs_test)
            del coeffs_test
    del Yi_test

 # Cropping the signal if the samples exceed the length of 2^12 
    if Zi_part2_train.shape[0] > (2**12) - 2*Q - 1:
        Zi_part2_train = Zi_part2_train[0:(2**12)-(2*Q)-1, :]
        if test_true:
            Zi_part2_test = Zi_part2_test[0:(2**12)-(2*Q)-1, :]      

    return Zi_part2_train, Zi_part2_test, zero_pad


def calculate_DT_score(Zi_part2, X_train, method): 
    percent = 0.8
    if method == 1: # (<) Variance of each sample (before ReLu) and then computing the mean among all samples
        var_s = np.std(Zi_part2, axis=1)
        score1 = np.std(var_s)  
        score2 = np.max(var_s)  
    elif method == 2: # (<) Correlation matrix (layer output - network input). Then compute the eigenvalues and get the index where the eigenvalues are the 90% of the total
        R_mat = np.corrcoef(Zi_part2, X_train, rowvar=True)
        R_mat[R_mat != R_mat] = 0 # To substitute NaNs for 0s
        R_mat = R_mat[len(X_train)-1:, :len(Zi_part2)]
        _, eigR, _ = np.linalg.svd(R_mat)
        eigR_accum = np.cumsum(eigR)/np.sum(eigR)
        eigR_accum = np.insert(eigR_accum, 0,0)
        idx = np.argmax(eigR_accum >= percent*eigR_accum[-1])
        score1 = 100 * idx / len(eigR_accum)
        score2 = eigR_accum[idx]
    return score1, score2

