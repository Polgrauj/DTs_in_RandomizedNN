import logging 
import argparse
from DRVFL import DRVFL
from DRVFL_rand import DRVFL_rand
from ELM import ELM
from ELM_rand import ELM_rand
from SSFN import SSFN
from SSFN_rand import SSFN_rand
from MyFunctions import *
from make_dataset_helper import *
import time
import json
import timeit
from timeit import default_timer as timer
import scipy

#   mnist   ,  ELM,    n1 = 1000   , eps_o = 10**(1)   , mu ~ 10**(-5)
#   ******* ,  RVFL,   n1 = 200*5  , eps_o = 10**(1)   , mu ~ 10**(4)
def define_parser():
    parser = argparse.ArgumentParser(description="Run \"randomized or deterministic transforms - based\" networks")
    parser.add_argument("--data", default="vowel", help="Input dataset available as the paper shows")
    parser.add_argument("--neural_net", default="SSFN", help="Neural network chosen: ELM, RVFL and SSFN available")
    parser.add_argument("--rand_net", default="Yes", help="Indicates if the neural network is tested with DT (No) or Rand (Yes)")
    parser.add_argument('--n_DT', type=int, default = 11, help='Number of desired Deterministic Transforms to be used in the LT part. Transforms are defined in the functin "DT_options" in "layer_iteration.py"')
    parser.add_argument("--MC_it", default=20, help="Number of montecarlo simulations")
    parser.add_argument('--alpha', type=int, default=2, help='User for optimization of the Output matrix on each layer (regularization parameter for ADMM)')
    parser.add_argument("--mu", type=float, default=1e3, help="Parameter for ADMM")
    parser.add_argument("--lam", type=float, default=1e2, help="Parameter for layer 0 of SSFN")
    parser.add_argument("--k_max", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--n1", type=int, default=1000, help="Max number of random nodes on each layer, when \"rand_net=Yes\"")
    parser.add_argument("--l_max", type=int, default=20, help="Number of layers")
    parser.add_argument('--method', type=int, default=2, help='Method used to chose unsupervised the deterministic transform: 1 (variance based) and 2 (singular values based)')
    parser.add_argument('--thr_var', type=float, default = 1e-7, help='Variance threshold for node reduction at each layer')
    parser.add_argument('--eta_l', type=float, default=0.1, help='Layers performance improvement threshold, to stop adding layers on SSFN')
    parser.add_argument("--eta_n", type=float, default=0.005, help="Threshold of nodes to stop adding nodes on SSFN when Rand=Yes")
    parser.add_argument("--delta", type=int, default=50, help="Number of random nodes to add once ion SSFN when Rand=Yes")
    args = parser.parse_args()
    return args

def define_dataset(args):
    if args.data == "letter":
        X_train, X_test, T_train,  T_test  = prepare_letter()
    elif args.data == "shuttle":
        X_train, X_test, T_train,  T_test  = prepare_shuttle()
    elif args.data == "mnist":
        X_train, X_test, T_train,  T_test  = prepare_mnist()
    elif args.data == "vowel":
        X_train, X_test, T_train,  T_test  = prepare_vowel()
    elif args.data == "glass":
        X_train, X_test, T_train,  T_test  = prepare_glass()
    elif args.data == "wine":
        X_train, X_test, T_train,  T_test  = prepare_wine()
    elif args.data == "iris":
        X_train, X_test, T_train,  T_test  = prepare_iris()
    elif args.data == "satimage":
        X_train, X_test, T_train,  T_test  = prepare_satimage()
    elif args.data == "caltech":
        X_train, X_test, T_train,  T_test  = prepare_caltech()
    elif args.data == "norb":
        X_train, X_test, T_train,  T_test  = prepare_norb()
    return X_train, X_test, T_train, T_test

def set_hparameters(args, neural_net):
    if neural_net == "ELM":
        hparameters     = {"data": args.data, "neural_net": args.neural_net, "MC_it":args.MC_it, "alpha": args.alpha, "mu": args.mu, "k_max": args.k_max, "n1": args.n1, "method": args.method, "thr_var":args.thr_var, "n_DT":args.n_DT}
    elif neural_net == "RVFL":
        hparameters   = {"data": args.data, "neural_net": args.neural_net, "MC_it":args.MC_it, "alpha": args.alpha, "mu": args.mu, "k_max": args.k_max, "n1": args.n1, "method": args.method, "thr_var":args.thr_var, "n_DT":args.n_DT, "l_max": args.l_max}
    elif neural_net == "SSFN":
        hparameters    = {"data": args.data, "neural_net": args.neural_net, "MC_it":args.MC_it, "alpha": args.alpha, "mu": args.mu, "k_max": args.k_max, "n1": args.n1, "method": args.method, "thr_var":args.thr_var, "n_DT":args.n_DT, "l_max": args.l_max, "lam": args.lam, "eta_l":args.eta_l, "eta_n":args.eta_n, "delta":args.delta}
    return hparameters

def main():
    data_path="./results/"

    args = define_parser()
    neural_net = args.neural_net
    rand_net = args.rand_net
    print("The dataset we use is "+ args.data)
    print("The neural network we use is: "+ neural_net)
    print("We use random matrix instance: "+rand_net)
    hparameters = set_hparameters(args, neural_net) # The dictionary of hyperparameters of the used network.


  # To plot the figures in the article with the saved values
    #load_nodes = np.load("Path to transforms similar to: mnist_method1_layernodes.npy")
    #load_layers = np.load("Path to layers similar to: mnist_method1_layerLTs.npy")
    #plot_with_colormap(load_nodes, load_layers, hparameters["data"])
    #load_nodes_M1 = np.load("Path to layer nodes of a network with Method 1: D:/Google Drive/Uni, work and documents/Projects/NNs with deterministic transforms/Simulation Parameters/SSFN/iris_method1_layernodes.npy")
    #load_nodes_M2 = np.load("Path to layer nodes of a network with Method 2: iris_method2_layernodes.npy")
    #plot_nodes_M1_M2(load_nodes_M1, load_nodes_M2, hparameters["data"])


    n_MC = args.MC_it # Number of montecarlo iterations
    acc_train_MC = []
    acc_test_MC = []
    if neural_net != "ELM":
        n_layers = []
        transforms = np.zeros([n_MC , hparameters["l_max"]])
        nodes = np.zeros([n_MC , hparameters["l_max"]])
    for i in range(0, n_MC):
        X_train, X_test, T_train, T_test = define_dataset(args) # Each column contains one sample
        #X_train = (X_train.T - np.mean(X_train, axis=1)).T # Used as a pre-processing in some datasets
        #X_test = (X_test.T - np.mean(X_test, axis=1)).T # Used as a pre-processing in some datasets
        if neural_net == "ELM":
            if rand_net == "Yes":
                train_accuracy, test_accuracy = ELM_rand(X_train, X_test, T_train, T_test, hparameters)
            else: 
                train_accuracy, test_accuracy = ELM(X_train, X_test, T_train, T_test, hparameters)
        elif neural_net == "RVFL":
            if rand_net == "Yes":
                train_accuracy, test_accuracy = DRVFL_rand(X_train, X_test, T_train, T_test, hparameters)
            else:
                node_lists, DT_lists, train_accuracy, test_accuracy = DRVFL(X_train, X_test, T_train, T_test, hparameters)
                transforms[i,:len(DT_lists)] = DT_lists
                nodes[i,:len(node_lists)] = node_lists
                n_layers.append(len(DT_lists))
        elif neural_net == "SSFN":
            if rand_net == "Yes":
                node_lists, train_acc, test_acc = SSFN_rand(X_train, X_test, T_train, T_test, hparameters)
                train_accuracy = train_acc[-1]
                test_accuracy = test_acc[-1]
            else:
                DTs = list(range(0, hparameters["n_DT"]+1))
                DTs = np.array([DTs,]*hparameters["l_max"]).transpose()
                node_lists, DT_lists, train_acc, test_acc = SSFN(X_train, T_train, X_test, T_test, hparameters, DTs)
                transforms[i,:len(DT_lists)] = DT_lists
                nodes[i,:len(node_lists)] = node_lists
                n_layers.append(len(DT_lists))
                train_accuracy = train_acc[-1]
                test_accuracy = test_acc[-1]
        acc_train_MC.append(train_accuracy)
        acc_test_MC.append(test_accuracy)
        print("Train accuracy cent: "+str(100*train_accuracy))
        print("Test accuracy cent: "+str(100*test_accuracy))
    acc_train_MC_mean = np.mean(acc_train_MC)
    acc_train_MC_dev = np.std(acc_train_MC)
    print("Train accuracy after {} MC simulations: {} + {}".format(n_MC, acc_train_MC_mean*100, acc_train_MC_dev*100))
    acc_test_MC_mean = np.mean(acc_test_MC)
    acc_test_MC_dev = np.std(acc_test_MC)
    print("Test accuracy after {} MC simulations: {} + {}".format(n_MC, acc_test_MC_mean*100, acc_test_MC_dev*100))
    if neural_net != "ELM" and rand_net != "Yes": # Save transforms and nodes of each layer when applying Deterministic transforms
        parameters_path = "./parameters/"
        create_directory(parameters_path)    
        save_parameters_det_MC(parameters_path, hparameters["data"], hparameters["method"], nodes, transforms, neural_net)


if __name__ == '__main__':
    main()