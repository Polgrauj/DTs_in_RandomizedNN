# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os
import math


def ELM_rand( X_train, X_test, T_train, T_test, ELM_hparameters):
    """Build Deep Random Vector Functional Link (DRVFL)"""
    # define variables
    P = X_train.shape[0]
    n1 = ELM_hparameters["n1"]
    
    # create a necessary directory
    parameters_path = "./parameters/"
    create_directory(parameters_path)

    N_train = X_train.shape[1]
    N_test = X_test.shape[1]
    X_train = np.concatenate((X_train, np.ones((1, N_train))), axis=0)
    X_test = np.concatenate((X_test, np.ones((1, N_test))), axis=0)
    Yi = X_train
    Yi_test=X_test
    P = Yi.shape[0]

    Ri = 2 * np.random.rand(n1, P) - 1

    Yi = activation(np.dot(Ri, Yi))
    Yi_test = activation(np.dot(Ri, Yi_test))
    Yi = Yi / np.linalg.norm(Yi, axis=0)
    Yi_test = Yi_test / np.linalg.norm(Yi_test, axis=0)

    Oi = ADMM_LS(Yi, T_train, ELM_hparameters)

    T_hati = np.dot(Oi, Yi) 
    T_hati_test = np.dot(Oi, Yi_test)

    train_accuracy = calculate_accuracy(T_hati, T_train)
    test_accuracy = calculate_accuracy(T_hati_test, T_test)
    
    return train_accuracy, test_accuracy
