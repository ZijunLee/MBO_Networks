import numpy as np
import random
from sklearn.metrics.cluster import adjusted_rand_score
from random import randrange

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

            print(stop_criterion,n)
        
        V_output = []
        for i in range(N):
            V_output.append(np.argmax(U_new[i,:]))
            
        return V_output

    #ARI = adjusted_rand_score(V_output)
    #print(ARI)

# test example: signed stochastic block model
# W is a matrix whose diagonal elements are 1 and the others are -1
diag_matrix = np.diag(np.full(20,2))
one_matrix = np.ones((20,20))
W_matrix = diag_matrix - one_matrix
#print(W_matrix)
# The degree matrix is a diagonal matrix whose entries are the sum of the rows of W_matrix
degree_matirx = np.diag(np.full(20,20))
#print(degree_matirx)
# Compute the positive parts of Laplacian matrix
laplacian_matrix = degree_matirx - W_matrix
#print(laplacian_matrix)

# Compute V*
N = laplacian_matrix.shape[0]
K = laplacian_matrix.shape[0]
m = K
V_star = mbo_modularity_eig(N, K, m, 0.1, laplacian_matrix, 10**(-7), 2)
print(V_star)

# Calculate ARI
ground_truth = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
ARI = adjusted_rand_score(V_star, ground_truth)
print(ARI)