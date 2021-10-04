import numpy as np
import os
import tensorflow as tf
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl


def save_list(my_list, parameters_path, data):
    with open(parameters_path + data+"_"+'n_lists.json','w') as f: 
        json.dump(my_list, f, ensure_ascii=False)


def save_dic(outputs, parameters_path, data, name):
    my_file = open(parameters_path + data + "_" + name + ".pkl", "wb")
    pickle.dump(outputs, my_file)
    my_file.close()


def load_dic( parameters_path, data, name):
    my_file = open(parameters_path + data + "_" + name + ".pkl", "rb")
    output = pickle.load(my_file)
    my_file.close()
    return output


def get_batch(Y, T, index, batchSize): 
    m = Y.shape[1]
    if batchSize < m:
        if index == (round(m/batchSize)-1):
            Y_batch = Y[:, index*batchSize:]
            T_batch = T[:, index*batchSize:]
        else:
            Y_batch = Y[:, index*batchSize:(index+1)*batchSize]
            T_batch = T[:, index*batchSize:(index+1)*batchSize]
    else:
        Y_batch = Y
        T_batch = T
    return Y_batch, T_batch


def compute_cost(S, Y):    
    # S = tf.nn.softmax(S, axis=1)
    # sum_cost = tf.math.reduce_mean(tf.keras.losses.MSE(tf.transpose(Y),tf.transpose(S)))
    # sum_cost = tf.math.reduce_mean(tf.keras.losses.squared_hinge(tf.transpose(Y),tf.transpose(S)))
    sum_cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf.transpose(a=Y)), logits=tf.transpose(a=S)))
    return sum_cost
    

def shuffle_data(Y, T):
    indices = tf.range(start=0, limit=Y.shape[1], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_Y = tf.gather(Y, shuffled_indices, axis=1)
    shuffled_T = tf.gather(T, shuffled_indices, axis=1)
    return shuffled_Y, shuffled_T


def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_v_values(Q):
    I = np.identity(Q, dtype=np.float32)
    concatenate_I = np.concatenate((I, -I), axis=0)
    return concatenate_I


def calculate_accuracy(T, T_hat):
    samples = T.shape[1]
    correct = np.argmax(T, axis = 0)
    predicted = np.argmax(T_hat, axis = 0)
    accuracy = np.sum([predicted == correct]) / samples
    return accuracy


def calculate_NME(T, T_hat):
    NME = 20 * np.log10(np.linalg.norm(T_hat - T, 'fro') / np.linalg.norm(T, 'fro'))
    return NME


def f_ReLu(x):
    return np.maximum(0, x)


def normalize_Z_SSFN_rand(tmp_Z, Q):
    Z_part1, Z_part2 = tmp_Z[:2*Q, :], tmp_Z[2*Q:, :]
    Z_part2 = Z_part2 / np.sum(Z_part2**2, axis=0, keepdims=True)**(1/2)
    Z = np.concatenate([Z_part1, Z_part2], axis=0)
    return Z

    
def activation(Z):
    Y = f_ReLu(Z)
    return Y


def SSFN_activation(x_1, x_2, part):
    if part == 2:
        y_2 = f_ReLu(x_2)
        return y_2
    elif part == "both":
        y_1 = f_ReLu(x_1)
        y_2 = f_ReLu(x_2)
        return y_1, y_2


def check_threshold(n_cost, o_cost, threshold):
    higher = (((o_cost - n_cost) / abs(o_cost)) >= threshold) # If false, stop iterate
    return higher


def normalization(prov_Z):
    Z = prov_Z / np.linalg.norm(prov_Z, axis=0)
    return Z


def prune_nodes(Z_part2, thr_var):
    var_n = np.var(Z_part2, axis=1)
    remove_n = np.where(np.abs(var_n) <= thr_var)
    return remove_n


def compute_random_nodes_transition(Q, n_lists, delta):
    random_nodes = np.array([0])
    n_lists = np.array(n_lists) - 2 * Q
    for idx, n in enumerate(n_lists):
        if idx == 0:
            el_random_nodes = np.array([nodes for nodes in range(0, n + 1, delta)])
        else:
            el_random_nodes = np.array([nodes for nodes in range(0, n + 1, delta)]) + sum(n_lists[:idx]) - 2 * Q * len(n_lists[:idx])
        random_nodes = np.append(random_nodes, el_random_nodes)
    return random_nodes


def plot_architecture(n_lists, Q, max_n, data_path, delta):
    random_nodes_lists = np.array(n_lists) - 2 * Q
    plt.figure(figsize=(10,10))
    plt.xlabel(xlabel="Layer Number")
    plt.ylabel(ylabel="Number of random nodes")
    plt.ylim(0, max_n)
    plt.scatter(x=range(1, len(random_nodes_lists)+1), y=random_nodes_lists)
    plt.savefig(data_path +'layer_num.png')


def plot_performance(xlabel, ylabel, random_nodes, train_performances, test_performances, data_path):
    plt.figure(figsize=(10, 10))
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.plot(random_nodes, train_performances, label="Train")
    plt.plot(random_nodes, test_performances, label="Test")
    plt.legend()
    plt.savefig(data_path + ylabel + '.png')


def plot_data(Q, n_lists, train_NME_lists, test_NME_lists, train_accuracy_lists, test_accuracy_lists, data, ssfn_hparameters):
    # define variables
    delta = ssfn_hparameters["delta"]
    max_n = ssfn_hparameters["max_n"]
    figure_path = "./figure/"
    data_path = figure_path + data +"/"

    # Create some directories for preservation
    create_directory(figure_path)
    create_directory(data_path)

    # The relation between layer number and number of nodes
    plot_architecture(n_lists, Q, max_n, data_path, delta)

    # The relations between number of random nodes and performances
    random_nodes = compute_random_nodes_transition(Q, n_lists, delta)
    plot_performance("Total number of random nodes","NME", random_nodes, train_NME_lists, test_NME_lists, data_path)
    plot_performance("Total number of random nodes" ,"Accuracy", random_nodes, train_accuracy_lists, test_accuracy_lists, data_path)


def save_parameters_det_MC(parameters_path, data, method, nodes_layers, LT_layers, NN):
    np.save(parameters_path+data+'_'+str(NN)+'_method'+str(method)+'_layernodes.npy', nodes_layers)
    np.save(parameters_path+data+'_'+str(NN)+'_method'+str(method)+'_layerLTs.npy', LT_layers)


def plot_with_colormap(nodes, LTs, data):
    figure_path = "./figure/"
    data_path = figure_path + data
    create_directory(figure_path)
    create_directory(data_path) 

  # Parameters initialization
    title = 'dRVFL DTs iris'
    MC = LTs.shape[0]
    n_LT = 12
    l_max = np.where(~nodes.any(axis=0))[0]
    #l_max = l_max[0]
    l_max = LTs.shape[1]
    LTs[nodes == 0] = -1

 # Linear Transforms plot
  # Figure initialization
    plot1 = plt.figure(figsize=(10, 10))
    ax1 = plot1.gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
  # Data definition
    y_appearances = np.zeros((n_LT, l_max))
    for l in range(0, l_max):
        for m in range(0, MC):
            if LTs[m,l] != -1:
                y_appearances[int(LTs[m, l]), l] += 1 
    x = list(range(1, l_max+1))
    x = np.tile(x, (n_LT, 1))
    y = list(range(0, n_LT))
    y = np.tile(y, (l_max, 1)).T
  # Colormap definition  
    c = y_appearances[::-1]
    c_map = plt.cm.hot.reversed() # Define the colormap
    c_maplist = [c_map(i) for i in range(c_map.N)] # extract all colors from the .jet map
    c_map = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', c_maplist, c_map.N)
    # define the bins and normalize
    bounds = np.linspace(0, 20, 21)
    norm = mpl.colors.BoundaryNorm(bounds, c_map.N)
  # Plot figure
    plt.scatter(x, y, c=y_appearances, cmap=c_map, s=600, norm=norm)
  # Plot configurations
    # create a second axes for the colorbar
    plt.grid(True, color='silver', linewidth=1)
    labels = ["DCT", "DST", "FWHT1", "FWHT2", "DHT", "DB4", "Haar", "sym2", "coif1", "bior1.3", "rbior1.1", "DB20"]
    ax1.tick_params(axis='both', labelsize = 26)
    ax1.set_yticklabels(labels)
    ax1.set_yticks(list(range(0, n_LT)))
    ax1.set_xticks(list(range(1, l_max+1)))
    ax1.set_title(title, fontsize = 38)
    plt.xlabel(xlabel='Layer Number', fontsize = 34)
    ax2 = plot1.add_axes([0.92, 0.1, 0.03, 0.8])
    ax2.tick_params(axis='y', labelsize = 20)
    mpl.colorbar.ColorbarBase(ax2, cmap=c_map, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax2.set_ylabel('DLTs appearances', fontsize=32)
  # Save figure
    figure_path = "./figure/"
    data_path = figure_path + data
    plt.savefig(data_path + '_DLTs.png', bbox_inches='tight')
    
    plt.show()

 # Nodes plot
  # Preparation
    plot2 = plt.figure(figsize=(10,10))
    ax3 = plot2.gca()
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    nodes_mean = []
    nodes_std = []
    for l in range(0, l_max):
        l_nodes = []
        MC_count = 0
        for m in range(0, MC):
            if nodes[m, l] != 0:
                l_nodes.append(nodes[m,l])
                MC_count += 1
        nodes_mean.append(np.mean(l_nodes, dtype=np.float32))
        nodes_std.append(np.std(l_nodes, dtype=np.float32))
        if l != l_max-1:
            plt.plot((l+1)*np.ones((MC_count)), l_nodes, '+b', markersize = 15, markeredgewidth=0.8)
    plt.plot((l+1)*np.ones((MC_count)), l_nodes, '+b', markersize = 15, markeredgewidth=0.8, label='Neurons instance')
  # Plot
    layers = list(range(1, l_max+1))
    plt.fill_between(layers, np.array(nodes_mean)-np.array(nodes_std), np.array(nodes_mean)+np.array(nodes_std), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label="Neurons std")
    plt.plot(layers, nodes_mean, 'k', label="Neurons mean")
    ax3.tick_params(axis='both', labelsize = 26)
    ax3.set_ylim(bottom=0)
    ax3.set_xticks(list(range(1, l_max+1)))
    ax3.set_title(title, fontsize = 38)
    plt.xlabel(xlabel='Layer Number', fontsize = 34)
    plt.ylabel(ylabel='Number of Neurons', fontsize = 34)
    plt.grid(True, color='silver', linewidth=1)
    plt.legend(loc='lower right', fontsize = 27)
  # Save figure
    figure_path = "./figure/"
    data_path = figure_path + data
    plt.savefig(data_path + '_nodes.png', bbox_inches='tight')    
    plt.show()


def plot_nodes_M1_M2(nodes_M1, nodes_M2, data):
    figure_path = "./figure/"
    data_path = figure_path + data
    create_directory(figure_path)
    create_directory(data_path) 

  # Parameters initialization
    title = 'SSFN Neurons iris'
    MC = nodes_M1.shape[0]
    l_max_M1 = np.where(~nodes_M1.any(axis=0))[0]
    l_max_M1 = l_max_M1[0]
    l_max_M2 = np.where(~nodes_M2.any(axis=0))[0]
    l_max_M2 = l_max_M2[0]
    l_max = max(l_max_M1, l_max_M2)
    #l_max = nodes_M1.shape[1]

 # Nodes plot
  # Preparation
    plot2 = plt.figure(figsize=(10,10))
    ax3 = plot2.gca()
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    nodes_mean_M1 = []
    nodes_std_M1 = []
    nodes_mean_M2 = []
    nodes_std_M2 = []
    for l in range(0, l_max_M1):
        l_nodes_M1 = []
        MC_count_M1 = 0
        for m in range(0, MC):
            if nodes_M1[m, l] != 0:
                l_nodes_M1.append(nodes_M1[m,l])
                MC_count_M1 += 1
        nodes_mean_M1.append(np.mean(l_nodes_M1, dtype=np.float32))
        nodes_std_M1.append(np.std(l_nodes_M1, dtype=np.float32))
        if l != l_max_M1-1:
            plt.plot((l+1)*np.ones((MC_count_M1)), l_nodes_M1, '+b', markersize = 15, markeredgewidth=0.8)
    plt.plot((l+1)*np.ones((MC_count_M1)), l_nodes_M1, '+b', markersize = 15, markeredgewidth=0.8, label='Neurons Method 1')
    for l in range(0, l_max_M2):
        l_nodes_M2 = []
        MC_count_M2 = 0
        for m in range(0, MC):
            if nodes_M2[m, l] != 0:
                l_nodes_M2.append(nodes_M2[m,l])
                MC_count_M2 += 1
        nodes_mean_M2.append(np.mean(l_nodes_M2, dtype=np.float32))
        nodes_std_M2.append(np.std(l_nodes_M2, dtype=np.float32))
        if l != l_max_M1-1:
            plt.plot((l+1)*np.ones((MC_count_M2)), l_nodes_M2, '+r', markersize = 15, markeredgewidth=0.8)
    plt.plot((l+1)*np.ones((MC_count_M2)), l_nodes_M2, '+r', markersize = 15, markeredgewidth=0.8, label='Neurons Method 2')

  # Plot
    layers_M1 = list(range(1, l_max_M1+1))
    layers_M2 = list(range(1, l_max_M2+1))
    plt.fill_between(layers_M1, np.array(nodes_mean_M1)-np.array(nodes_std_M1), np.array(nodes_mean_M1)+np.array(nodes_std_M1), alpha=0.5, edgecolor='#06A2D3', facecolor='#B8E1EE')
    plt.fill_between(layers_M2, np.array(nodes_mean_M2)-np.array(nodes_std_M2), np.array(nodes_mean_M2)+np.array(nodes_std_M2), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.plot(layers_M1, nodes_mean_M1, 'b')
    plt.plot(layers_M2, nodes_mean_M2, 'r')
    ax3.tick_params(axis='both', labelsize = 26)
    ax3.set_ylim(bottom=0)
    ax3.set_xticks(list(range(1, l_max+1)))
    ax3.set_title(title, fontsize = 38)
    plt.xlabel(xlabel='Layer Number', fontsize = 34)
    plt.ylabel(ylabel='Number of Neurons', fontsize = 34)
    plt.grid(True, color='silver', linewidth=1)
    plt.legend(loc='upper right', fontsize = 27)
  # Save figure
    figure_path = "./figure/"
    data_path = figure_path + data
    plt.savefig(data_path + '_nodes_M1_M2.png', bbox_inches='tight')    
    plt.show()