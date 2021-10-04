# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os
from layer_iteration import *


def DRVFL( X_train, X_test, T_train, T_test, DRVFL_hparameters):
    """Build Deep Random Vector Functional Link (DRVFL)"""
    # define variables
    P = X_train.shape[0]
    Q = T_train.shape[0]
    l_max = DRVFL_hparameters["l_max"]
    method = DRVFL_hparameters["method"]
    DTs = list(range(0, DRVFL_hparameters["n_DT"]+1))
    DTs = np.array([DTs,]*1).transpose()

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
    Pi= Yi.shape[0]

    D = Yi
    D_test = Yi_test

    T_hati_all = np.empty([T_train.shape[0], T_train.shape[1], l_max])
    T_hati_test_all = np.empty([T_test.shape[0], T_test.shape[1], l_max])

    for layer in range(1, l_max + 1):
        DT_mat = DTs[:,0]
        DT_num = len(DT_mat)
        if DT_num > 1:
            for DT_it in DT_mat:
                Zi_train, _, zero_pad = DT_options(Yi, [], DT_it, Q, False)
                remove_n = prune_nodes(activation(normalization(Zi_train)), DRVFL_hparameters["thr_var"]) # Prune nodes
                Zi_train = np.delete(Zi_train, remove_n, axis=0) # Prune nodes
                Zi_train = normalization(Zi_train)
                if Zi_train.any(): # May happen all nodes are pruned
                    DT_score1_it, DT_score2_it = calculate_DT_score(Zi_train, X_train, method)
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

        Zi_train, Zi_test, zero_pad = DT_options(Yi, Yi_test, DT_lth, Q, True)
        remove_n = prune_nodes(activation(normalization(Zi_train)), DRVFL_hparameters["thr_var"]) # Prune nodes
        Zi_train = np.delete(Zi_train, remove_n, axis=0) # Prune nodes
        if not Zi_train.any(): # May happen all nodes are pruned --> Stop without this layer
            break
        Zi_test = np.delete(Zi_test, remove_n, axis=0) # Prune nodes
        Zi_train = normalization(Zi_train)
        Zi_test = normalization(Zi_test)

        Yi = activation(Zi_train)
        Yi_test = activation(Zi_test)
        #Yi = Yi / np.linalg.norm(Yi, axis=0) # Sometimes is useful to arrest energy increase through layers
        #Y_test = Yi_test / np.linalg.norm(Yi_test, axis=0) # Sometimes is useful to arrest energy increase through layers
        del Zi_train, Zi_test, zero_pad

        D = np.concatenate((D, Yi), axis=0)
        D_test = np.concatenate((D_test, Yi_test), axis=0)

        print("Layer iteration: "+str(layer))
        n_nodes = Yi.shape[0]
        node_lists.append(n_nodes)
        LT_lists.append(int(DT_lth))

    Oi = ADMM_LS(D, T_train, DRVFL_hparameters) # Output matrix
        
    T_hati = np.dot(Oi, D) 
    T_hati_test = np.dot(Oi, D_test)

    train_accuracy = calculate_accuracy(T_hati, T_train)
    test_accuracy = calculate_accuracy(T_hati_test, T_test)

    print("Deterministic Transform chosen: {}".format(LT_lists))
    print("Nodes added: {}".format(node_lists))
    
    return node_lists, LT_lists, train_accuracy, test_accuracy

