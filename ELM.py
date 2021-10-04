# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os
import math
from layer_iteration import *


def ELM( X_train, X_test, T_train, T_test, ELM_hparameters):
    """Build Deep Random Vector Functional Link (DRVFL)"""
    # define variables
    Q = T_train.shape[0]
    DTs = list(range(0, ELM_hparameters["n_DT"]+1))
    DTs = np.array([DTs,]*1).transpose()
    method = ELM_hparameters["method"]
    
    # Initializations
    node_lists = []
    LT_lists = []

    # create a necessary directory
    parameters_path = "./parameters/"
    create_directory(parameters_path)

    N_train = X_train.shape[1]
    N_test = X_test.shape[1]
    X_train=np.concatenate((X_train, np.ones((1,N_train))), axis=0)
    X_test=np.concatenate((X_test, np.ones((1,N_test))), axis=0)
    Yi=X_train
    Yi_test=X_test

    test_flag = False
    DT_mat = DTs[:,0]
    DT_num = len(DT_mat)
    if DT_num > 1:
        for DT_it in DT_mat:
            Zi_train, _, zero_pad = DT_options(Yi, [], DT_it, Q, test_flag)
            remove_n = prune_nodes(activation(Zi_train), ELM_hparameters["thr_var"]) # Prune nodes
            Zi_train = np.delete(Zi_train, remove_n, axis=0) # Prune nodes
            Zi_train = normalization(Zi_train)
            DT_score1_it, DT_score2_it = calculate_DT_score(Zi_train, X_train, method) # Compute score "sc_1"
            if DT_it == DT_mat[0]: 
                DT_score1 = DT_score1_it
            if (DT_it == DT_mat[0]) or (DT_score1_it < DT_score1):
                DT_score1 = DT_score1_it
                DT_score2 = DT_score2_it
                DT_lth = DT_it
            elif DT_score1_it == DT_score1:
                if DT_score2_it > DT_score2:
                    DT_score1 = DT_score1_it
                    DT_score2 = DT_score2_it
                    DT_lth = DT_it
    else:  
        DT_lth = DT_mat

    test_flag = True
    Zi_train, Zi_test, zero_pad = DT_options(Yi, Yi_test, DT_lth, Q, test_flag)
    remove_n = prune_nodes(activation(Zi_train), ELM_hparameters["thr_var"]) # Prune nodes
    Zi_train = np.delete(Zi_train, remove_n, axis=0) # Prune nodes
    Zi_test = np.delete(Zi_test, remove_n, axis=0) # Prune nodes
    Zi_train = normalization(Zi_train)
    Zi_test = normalization(Zi_test)
    Yi = activation(Zi_train)
    Yi_test = activation(Zi_test)
    del Zi_train, Zi_test, zero_pad   

    n_nodes = Yi.shape[0]
    node_lists.append(n_nodes)
    LT_lists.append(int(DT_lth))

    Oi = ADMM_LS(Yi, T_train, ELM_hparameters) # Output matrix

    T_hati = np.dot(Oi, Yi) 
    T_hati_test = np.dot(Oi, Yi_test)

    train_accuracy = calculate_accuracy(T_hati, T_train)
    test_accuracy = calculate_accuracy(T_hati_test, T_test)
    
    print("Deterministic Transform chosen: {}".format(LT_lists))
    print("Nodes added: {}".format(node_lists))

    return train_accuracy, test_accuracy

