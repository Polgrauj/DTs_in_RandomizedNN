from MyFunctions import *
from MyOptimizers import *
import numpy as np
from layer_iteration import *


def SSFN(X_train, T_train, X_test, T_test, ssfn_parameters, DTs):
   # Variable Definitions
   l_max = ssfn_parameters["l_max"]
   eta_l = ssfn_parameters["eta_l"]
   data = ssfn_parameters["data"]
   layers = DTs.shape[0] 
   method = ssfn_parameters["method"]

   # Parameters Definitions
   P = X_train.shape[0]
   Q = T_train.shape[0]
   VQ = np.concatenate([np.identity(Q), -np.identity(Q)], axis=0).astype(np.float32)

   # Lists Initializations
   node_lists = []
   outputs = {}
   accuracy_train_lists = []
   accuracy_test_lists = []
   NME_train_lists = []
   NME_test_lists = []    
   LT_lists = []

   # Directory Definition
   parameters_path = "./parameters/"
   create_directory(parameters_path)

   # Adding bias to the input
   Y_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)
   Y_test = np.concatenate((X_test, np.ones((1, X_test.shape[1]))), axis=0)

   # First Layer Block (Regularized Least Squares)
   W_ls, NME_train, NME_test, T_hat_train, T_hat_test, accuracy_train, accuracy_test = Regularised_LS(Y_train, T_train, \
         Y_test, T_test, ssfn_parameters["lam"])

   # Preserve the optimized cost and parameters
   O = W_ls
   outputs["W_ls"] = W_ls
   accuracy_train_lists.append(accuracy_train)
   accuracy_test_lists.append(accuracy_test)
   NME_train_lists.append(NME_train)
   NME_test_lists.append(NME_test)     
   o_NME = NME_train # Initialize the old NME to compare when adding layers

   del X_test, W_ls

   # Iterate among layers (ADMM)
   for l_layer in range(0, layers):
      Zi_part1_train = np.dot(VQ, T_hat_train)
      Zi_part1_test = np.dot(VQ, T_hat_test)

      if Y_train.shape[0] > (2**12):
         Y_train = Y_train[0:(2**12), :]
         Y_test = Y_test[0:(2**12), :]

      # First the network has to be trained using only the training signal
      # First, to chose the transform, test data is not needed, reducing training computational cost 
      test_flag = False
      DT_mat = DTs[:,l_layer]
      DT_num = len(DT_mat)
      if DT_num > 1:
         for DT_it in DT_mat:
            Zi_part2_train, _, zero_pad = DT_options(Y_train, [], DT_it, Q, test_flag)
            remove_n = prune_nodes(SSFN_activation([], normalization(Zi_part2_train), 2), ssfn_parameters["thr_var"]) # Prune nodes
            Zi_part2_train = np.delete(Zi_part2_train, remove_n, axis=0) # Prune nodes
            Zi_part2_train = normalization(Zi_part2_train)
            DT_score1_it, DT_score2_it = calculate_DT_score(Zi_part2_train, X_train, method) # Compute score "sc_1"
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

      # Now that the Deterministic Transform is chosen, test data is also used
      test_flag = True
      Zi_part2_train, Zi_part2_test, zero_pad = DT_options(Y_train, Y_test, DT_lth, Q, test_flag)
      remove_n = prune_nodes(SSFN_activation([], normalization(Zi_part2_train), 2), ssfn_parameters["thr_var"]) # Prune nodes
      Zi_part2_train = np.delete(Zi_part2_train, remove_n, axis=0) # Prune nodes
      Zi_part2_test = np.delete(Zi_part2_test, remove_n, axis=0) # Prune nodes
      O, Y_train, T_hat_train, Y_test, T_hat_test, NME_train, NME_test, accuracy_train, accuracy_test = \
         SSFN_Signal_Flow(Zi_part1_train, Zi_part2_train, T_train, Zi_part1_test, Zi_part2_test, T_test, ssfn_parameters, Q) # Signal flow within SSFN  
        
      del Zi_part2_train, Zi_part2_test, Zi_part1_train, Zi_part1_test, zero_pad

      if not check_threshold(NME_train, o_NME, eta_l): # Check cost-layer with the threshold to stop if reduction is not significative enough
         break

      n_nodes = Y_train.shape[0]

      # Preserve optimized parameters and resulting signals
      node_lists.append(n_nodes)
      outputs["O"+str(l_layer)] = O
      outputs["remove_n"+str(l_layer)] = remove_n
      del O, remove_n

      # Preserve the lth layers performance
      accuracy_train_lists.append(accuracy_train)
      accuracy_test_lists.append(accuracy_test)
      NME_train_lists.append(NME_train)
      NME_test_lists.append(NME_test) 
      LT_lists.append(int(DT_lth))     
        
      # Updating the old MSE to compare later with the threshold
      o_NME = NME_train

   # save_SSFN_parameters(outputs, parameters_path, data, node_lists, accuracy_train_lists, accuracy_test_lists, NME_train_lists, NME_test_lists, LT_lists)

   return node_lists, LT_lists, accuracy_train_lists, accuracy_test_lists



def SSFN_Signal_Flow(Zi_part1_train, Zi_part2_train, T_train, Zi_part1_test, Zi_part2_test, T_test, ssfn_parameters, Q):
 # Normalization step
    Zi_part2_train = normalization(Zi_part2_train)
    Zi_part2_test = normalization(Zi_part2_test)

 # Activation step
    Yi_train_part1, Yi_train_part2 = SSFN_activation(Zi_part1_train, Zi_part2_train, "both")
    del Zi_part1_train, Zi_part2_train
    Yi_test_part1, Yi_test_part2 = SSFN_activation(Zi_part1_test, Zi_part2_test, "both")
    del Zi_part1_test, Zi_part2_test    

    Yi_train = np.concatenate([Yi_train_part1, Yi_train_part2], axis=0)  
    del Yi_train_part1, Yi_train_part2
    Yi_test = np.concatenate([Yi_test_part1, Yi_test_part2], axis=0)
    del Yi_test_part1, Yi_test_part2

 # Adding bias component 
    Yi_train = np.concatenate([Yi_train, np.ones((1, T_train.shape[1]))], axis=0) 
    Yi_test = np.concatenate([Yi_test, np.ones((1, T_test.shape[1]))], axis=0) 

 # Solve ADMM and compute output matrix
    Oi = ADMM_LS(Yi_train, T_train, ssfn_parameters)
    T_hati_train = np.dot(Oi, Yi_train)
    T_hati_test = np.dot(Oi, Yi_test)

 # Compute errors and accuracies
    NME_train = calculate_NME(T_train, T_hati_train)
    accuracy_train = calculate_accuracy(T_train, T_hati_train)
    NME_test = calculate_NME(T_test, T_hati_test)
    accuracy_test = calculate_accuracy(T_test, T_hati_test)

    return Oi, Yi_train, T_hati_train, Yi_test, T_hati_test, NME_train, NME_test, accuracy_train, accuracy_test



def save_SSFN_parameters(outputs, parameters_path, data, n_lists, accuracy_train, accuracy_test, NME_train, NME_test, LT_list):
    for key, value in outputs.items():
        np.save(parameters_path + data +'_'+key + ".npy", value)

    with open(parameters_path + data + "_" + 'n_lists.json', 'w') as f: 
        json.dump(n_lists, f, ensure_ascii=False)
    with open(parameters_path + data + "_" + 'accuracy_train.json', 'w') as f: 
        json.dump(accuracy_train, f, ensure_ascii=False)
    with open(parameters_path + data + "_" + 'accuracy_test.json', 'w') as f: 
        json.dump(accuracy_test, f, ensure_ascii=False)        
    with open(parameters_path + data + "_" + 'NME_train.json', 'w') as f: 
        json.dump(NME_train, f, ensure_ascii=False)
    with open(parameters_path + data + "_" + 'NME_test.json', 'w') as f: 
        json.dump(NME_test, f, ensure_ascii=False)
    with open(parameters_path + data + "_" + 'LT_list.json', 'w') as f: 
        json.dump(LT_list, f, ensure_ascii=False)