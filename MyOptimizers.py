import numpy as np
from MyFunctions import *

def Regularised_LS(X_train, T_train, X_test, T_test, lambda_ls):
    P = X_train.shape[0]
    m = X_train.shape[1]

    if P < m:
        W_ls = np.dot(np.dot(T_train, X_train.T), np.linalg.inv(np.dot(X_train, X_train.T) + lambda_ls*np.eye(P)))
    else:
        W_ls = np.dot(T_train, np.linalg.inv(np.dot(X_train.T, X_train) + lambda_ls*np.eye(m))).dot(X_train.T)
    W_ls = np.float32(W_ls)

    T_train_hat = np.dot(W_ls, X_train)
    T_test_hat = np.dot(W_ls, X_test)
    NME_train = calculate_NME(T_train, T_train_hat)
    NME_test = calculate_NME(T_test, T_test_hat)
    accuracy_train = calculate_accuracy(T_train, T_train_hat)
    accuracy_test = calculate_accuracy(T_test, T_test_hat)

    return W_ls, NME_train, NME_test, T_train_hat, T_test_hat, accuracy_train, accuracy_test


def ADMM_LS(Y_train, T_train, hparameters):
    n = Y_train.shape[0]
    Q = T_train.shape[0]
    mu = hparameters["mu"]
    k_max = hparameters["k_max"]
    alpha = hparameters["alpha"]
    Lambdas = np.zeros((Q, n))
    Q_mat = np.zeros((Q,n))
    TY_mat = np.dot(T_train, Y_train.T)
    YY_mat = np.dot(Y_train, Y_train.T)
    aux = YY_mat + 1/mu * np.eye(n)
    epsilon = alpha * np.sqrt(2*Q)
    for _ in range(k_max):
        O = np.dot(TY_mat + 1/mu * (Q_mat + Lambdas), np.linalg.inv(aux))
    # Projection for ADMM
        Q_mat = O - Lambdas
        norm_Q = np.linalg.norm(Q_mat, 'fro')
        if norm_Q > epsilon:
                Q_mat = Q_mat * epsilon / norm_Q
    # Lagrange Multipliers update    
        Lambdas = Lambdas + Q_mat - O
    
    return O