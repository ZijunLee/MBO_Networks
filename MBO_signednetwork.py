import numpy as np
import random
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.linalg import sqrtm
from random import randrange
import os
import sys

from Function_test import A_matrix
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import scipy.sparse as ss
import numpy.random as rnd
import math
import networkx as nx
from signet.utils import sqrtinvdiag
from signet.cluster import Cluster
from signet.block_models import SSBM
from sklearn.metrics import adjusted_rand_score


""" Parameters
    -----------
    N : scalar,
        the number of nodes
    K : scalar,
        the number of clusters (note: N >> K)
    m : scalar,
        the number of eigenvectors and eigenvalues 
    inner_step_count : int
        for MBO_modularity, Number of times the scheme performs
    dt : float
        stepsize for scheme
    tol : scalar, 
        stopping criterion for iteration
    laplacian_matrix_ : array-like, shape (n_samples, n_samples)
        signed Laplacian matrix
    ground_truth : ndarray, (n_samples,).{0...K-1} labels.
        labels corresponding to the raw data  
"""

def mbo_modularity_eig(N,K,m,dt,laplacian_matrix_, tol ,inner_step_count): # inner stepcount is actually important! and can't be set to 1...
    
    if N<K:
        print("Wrong input, N should larger or equal than K.")
    
    else:
        # Initialize parameters
        # eigendecomposition of the Laplacian
        EigVal, EigVec = np.linalg.eig(laplacian_matrix_)
        X_m = EigVec[0:m,:]
        Val_m = np.diag(EigVal[0:m])
        identity_matrix_m = np.diag(np.full(m,1))
        B_m = (X_m.T).dot(np.linalg.inv(identity_matrix_m + dt*Val_m)).dot(X_m)
        U_init = np.zeros((N,K))
        for i in range(N):
            k = randrange(K-1)
            U_init[i,k] = 1
        
        # Ensure each cluster has at least one node
        # Generate data list that store one node for each cluster
        K_force = random.sample(range(K),K)
        
        # Random select rows with the amount of clusters
        K_row = random.sample(range(N),K)
        for j in range(K):

            # Force the selected row to be zero vector
            U_init[K_row[j],:] = np.zeros(K)

            # Store the data list determined node to this cluster
            U_init[K_row[j],K_force[j]] = 1

        # print(U_init)
        # Perform MBO scheme
        n = 0
        U_old = U_init.copy()
        stop_criterion = 1
        while (stop_criterion > tol):  # Diffusion
            U_half_old = U_old
            #print(U_old)
            for s in range(inner_step_count):
                U_half_new = np.dot(B_m, U_half_old)
                U_half_old = U_half_new
                s = s + 1
            
            #U_half_new = np.dot(np.linalg.matrix_power((X_m.T).dot(np.linalg.inv(identity_matrix_m + (dt/inner_step_count)*Val_m)).dot(X_m), inner_step_count), U_half_old)
            #print(U_half_new)
                       
            U_new = np.zeros((N,K))
            for i in range(N):   # Thresholding
                k = np.argmax(U_half_new[i,:])
                U_new[i,k] = 1

            # Stop criterion
            Ui_diff = []
            Ui_max = []
            for i in range(N):    
                Ui_diff.append((np.linalg.norm(U_new[i,:] - U_old[i,:]))**2)
                Ui_max.append((np.linalg.norm(U_new[i,:]))**2)
                
            max_diff = max(Ui_diff)
            max_new = max(Ui_max)
            stop_criterion = max_diff/max_new

            #print(U_new)
            U_old = U_new

            n = n + 1

            #print(stop_criterion,n)
        
        V_output = []
        for i in range(N):
            V_output.append(np.argmax(U_new[i,:]))
            
        return V_output



# test example: signed stochastic block model

# generate a random graph with community structure by the signed stochastic block model 
def SSBM(N, K):
    s_matrix = -np.ones((N,N))
    randomlist = random.sample(range(1, N), K-1)
    randomlist.append(0)
    randomlist.append(N)
    randomlist.sort()
    #print(randomlist)

    randomquantity = []
    for quantity in range(len(randomlist)-1):
        randomquantity.append(randomlist[quantity+1]-randomlist[quantity])
    #print(randomquantity)

    for randomnumber in range(len(randomlist)):
        for i in range(randomlist[randomnumber-1], randomlist[randomnumber]):
            for j in range(randomlist[randomnumber-1], randomlist[randomnumber]):
                s_matrix[i][j] = 1
    #print(s_matrix)

    ground_truth = []
    for gt in range(len(randomquantity)):
        ground_truth.extend([gt for y in range(randomquantity[gt])])
    #print(ground_truth)

    ground_truth_v2 = []
    for gt in range(len(randomquantity)):
        ground_truth_v2.extend([randomquantity[gt] for y in range(randomquantity[gt])])
    #print(ground_truth_v2)

    return s_matrix, ground_truth

def data_generator(s_matrix, noise, sparsity):
    A_init_matrix = s_matrix
    N = s_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            if i >= j:
                A_init_matrix[i][j] = 0
            if i < j:
                elements = [A_init_matrix[i][j], 0, -A_init_matrix[i][j]]
                probabilities = [(1- noise)*sparsity, 1-sparsity, noise*sparsity]
                A_init_matrix[i][j] = np.random.choice(elements, 1, p=probabilities)
    A_matrix = A_init_matrix + A_init_matrix.T - np.diag(np.diag(A_init_matrix))

    # calculate the signed laplacian matrix / symmetric normalized laplacian
    A_absolute_matrix = np.abs(A_matrix)
    sum_row = np.sum(A_absolute_matrix,axis=1).tolist()
    deg_diag_m = np.diag(sum_row)
    deg_inverse_m = sqrtm(deg_diag_m)
    laplacian_m = deg_diag_m - A_matrix
    Dbar_half = np.sqrt(deg_inverse_m)
    laplacian_sym = np.diag(np.full(N,1)) - (Dbar_half).dot(A_matrix).dot(Dbar_half)

    return laplacian_sym
        

# Parameter setting
N = 1200
K = 2
noise = 0.05
sparsity = 0.02
m = K
dt = 0.1
tol = 10**(-7)
inner_step_count = 3

s_matrix, ground_truth = SSBM(N,K)
laplacian_matrix = data_generator(s_matrix, noise, sparsity)
V_output = mbo_modularity_eig(N, K, m, dt, laplacian_matrix, tol, inner_step_count)
ARI = adjusted_rand_score(V_output,ground_truth)
print(ARI)