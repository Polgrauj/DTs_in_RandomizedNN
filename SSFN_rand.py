# -*- coding: utf-8 -*-

import numpy as np
import logging
from MyFunctions import *
from MyOptimizers import *
import json
import os


def SSFN_rand(X_train, X_test, T_train, T_test, ssfn_hparameters):
    """Build self size-estimating Feed-forward network(SSFN)"""
    # define variables
    Q = T_train.shape[0]
    l_max = ssfn_hparameters["l_max"]
    eta_l = ssfn_hparameters["eta_l"]
    data = ssfn_hparameters["data"]
    
    # initialize collections
    n_lists = []
    outputs = {}
    train_accuracy_lists = []
    test_accuracy_lists = []
    train_NME_lists = []
    test_NME_lists = []
    
    # create a necessary directory
    parameters_path = "./parameters/"
    create_directory(parameters_path)

    W_ls, train_NME, test_NME, T_hat_train, T_hat_test, train_accuracy, test_accuracy = \
        Regularised_LS(X_train, T_train, X_test, T_test, ssfn_hparameters["lam"])

    # preserve the optimized cost and parameter
    outputs["W_ls"] = W_ls
    train_accuracy_lists.append(train_accuracy)
    test_accuracy_lists.append(test_accuracy)
    train_NME_lists.append(train_NME)
    test_NME_lists.append(test_NME)

    old_train_NME = train_NME # initialize old train NME
    for lth_layer in range(1, l_max + 1):
        W, Yi_train, Yi_test, nodes, O, R, el_train_accuracy_lists, \
        el_train_NME_lists, el_test_accuracy_lists, el_test_NME_lists = \
                 optimize_each_layer(X_train, X_test, T_train, T_test, outputs, n_lists, lth_layer, ssfn_hparameters)

        current_train_NME = el_train_NME_lists[-1]
        if not check_threshold(current_train_NME, old_train_NME, eta_l):
            # break condition
            break
        
        # preserve optimized parameters
        n_lists.append(nodes)
        outputs["W" +str(lth_layer)] = W
        outputs["R" + str(lth_layer)] = R
        outputs["O"+str(lth_layer)] = O
        outputs["Y_train" +str(lth_layer)] = Yi_train
        outputs["Y_test"+str(lth_layer)] = Yi_test

        # preserve the lth layers performance
        train_accuracy_lists.extend(el_train_accuracy_lists)
        test_accuracy_lists.extend(el_test_accuracy_lists)
        train_NME_lists.extend(el_train_NME_lists)
        test_NME_lists.extend(el_test_NME_lists)

        # update old train NME
        old_train_NME = current_train_NME

    # save_SSFN_rand_parameters(outputs, parameters_path, data, n_lists)

    return n_lists, train_accuracy_lists, test_accuracy_lists



def optimize_each_layer(X_train, X_test, T_train, T_test, outputs, n_lists, lth_layer, ssfn_parameters):
    """Build the structure on l'th layer"""
    P = X_train.shape[0]
    Q = T_train.shape[0]
    eta_n = ssfn_parameters["eta_n"]
    n_max = ssfn_parameters["n1"]
    k_max = ssfn_parameters["k_max"]
    delta = ssfn_parameters["delta"]
    
    # accuracy and NME on each layer
    el_train_accuracy_lists = []
    el_test_accuracy_lists = []
    el_train_NME_lists = []
    el_test_NME_lists = []
    
    # set the initialized parameter
    nodes = 2 * Q
    prev_nodes = n_lists[lth_layer-2] if lth_layer != 1 else P
    cache_n = {}
    V = create_v_values(Q)
    
    # set up  parameter on previous layer
    prev_O = outputs["O"+str(lth_layer-1)] if lth_layer != 1 else outputs["W_ls"]
    prev_Y_train = outputs["Y_train" + str(lth_layer -1)] if lth_layer != 1 else X_train
    prev_Y_test = outputs["Y_test" + str(lth_layer -1)] if lth_layer != 1 else X_test
    
    max_iter_num = int(n_max/delta) + 1
    for iter_j in range(max_iter_num):
        nodes = 2 * Q + iter_j * delta
        # set the R and W values
        if iter_j == 0:
            R = np.array([], dtype=np.float32)
            W = np.dot(V, prev_O)
        else:
            R = 2 * np.random.rand(nodes- 2 * Q, prev_nodes) - 1 if iter_j == 1 else np.concatenate([R, 2 * np.random.rand(delta, prev_nodes) - 1], axis=0)
            W = np.concatenate([np.dot(V, prev_O), R], axis=0)
        Z_train_temp = np.dot(W, prev_Y_train)
        Zi_train = normalize_Z_SSFN_rand(Z_train_temp, Q)
        Yi_train = activation(Zi_train)
        Oi = ADMM_LS(Yi_train, T_train, ssfn_parameters)
        Si_train = np.dot(Oi, Yi_train) 

        # for test data
        Z_test_temp = np.dot(W, prev_Y_test)
        Zi_test = normalize_Z_SSFN_rand(Z_test_temp, Q)
        Yi_test = activation(Zi_test)
        Si_test = np.dot(Oi, Yi_test)
        cNME = calculate_NME(T_train, Si_train)
        
        # break condition 
        if iter_j != 0 and not check_threshold(cNME, oNME, eta_n):
            Yi_train = cache_n["Y"]
            Yi_test = cache_n["Y_test"]
            W = cache_n["W"]
            nodes = cache_n["nodes"]
            R = cache_n["R"]
            Oi = cache_n["O"]
            break
        
        # update previous result
        cache_n["W"]  = W
        cache_n["R"]  = R
        cache_n["Y"] = Yi_train
        cache_n["Y_test"] = Yi_test
        cache_n["nodes"] = nodes
        cache_n["O"] = Oi
        cache_n["NME"] = cNME
        oNME = cNME
        
        # Accuracy train
        train_accuracy = calculate_accuracy(Si_train, T_train)
        train_NME = cNME

        #  Accuracy test
        test_NME = calculate_NME(T_test, Si_test)
        test_accuracy = calculate_accuracy(Si_test, T_test)

        # preserve performance
        el_train_NME_lists.append(train_NME)
        el_train_accuracy_lists.append(train_accuracy)
        el_test_NME_lists.append(test_NME)
        el_test_accuracy_lists.append(test_accuracy)

    return W, Yi_train, Yi_test, nodes, Oi, R, el_train_accuracy_lists, el_train_NME_lists, el_test_accuracy_lists, el_test_NME_lists



def save_SSFN_rand_parameters(outputs, parameters_path, data, n_lists):
    for key, value in outputs.items():
        np.save(parameters_path + data +'_'+key + ".npy", value)

    with open(parameters_path + data+"_"+'n_lists.json','w') as f: 
        json.dump(n_lists, f, ensure_ascii=False)