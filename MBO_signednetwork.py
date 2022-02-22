import numpy as np
import random
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.linalg import sqrtm
from random import randrange
import os
import sys

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

def mbo_modularity_own(N,K,m,dt,laplacian_matrix_, tol ,inner_step_count): # inner stepcount is actually important! and can't be set to 1...
    
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

    # calculate the signed laplacian matrix
    A_absolute_matrix = np.abs(A_matrix)
    sum_row = np.sum(A_absolute_matrix,axis=1).tolist()
    deg_diag_m = np.diag(sum_row)
    laplacian_m = deg_diag_m - A_matrix

    # calculate Dbar^(-1/2)
    sum_list = []
    for element in sum_row:
        if element == 0:
            sum_list.append(element)
        else:
            sum_list.append(1.0/(np.sqrt(element)))
    
    deg_inverse_m = np.diag(sum_list)
    #pseudo_inv_deg_m = np.linalg.pinv(deg_diag_m)
    #Dbar_half = np.sqrt(pseudo_inv_deg_m)
    
    # calculate the symmeytic normalized laplacian
    laplacian_sym = (deg_inverse_m).dot(laplacian_m).dot(deg_inverse_m)
    #laplacian_sym_pseu_inv = (Dbar_half).dot(laplacian_m).dot(Dbar_half)

    return laplacian_sym
        
def adjacancy_to_laplacian(A_matrix):
    # calculate the signed laplacian matrix
    A_absolute_matrix = np.abs(A_matrix)
    sum_row = np.sum(A_absolute_matrix,axis=1).tolist()
    deg_diag_m = np.diag(sum_row)
    laplacian_m = deg_diag_m - A_matrix

    # calculate Dbar^(-1/2)
    #sum_list = []
    #for element in sum_row:
    #    if element == 0:
    #        sum_list.append(element)
    #    else:
    #        sum_list.append(1.0/(np.sqrt(element)))
    
    #deg_inverse_m = np.diag(sum_list)
    #pseudo_inv_deg_m = np.linalg.pinv(deg_diag_m)
    #Dbar_half = np.sqrt(pseudo_inv_deg_m)
    
    # calculate the symmeytic normalized laplacian
    #laplacian_sym = (deg_inverse_m).dot(laplacian_m).dot(deg_inverse_m)
    #laplacian_sym_pseu_inv = (Dbar_half).dot(laplacian_m).dot(Dbar_half)

    return laplacian_m

# Parameter setting
N = 34
K = 2
noise = 0.18
sparsity = 0.02
m = K
dt = 0.1
tol = 10**(-7)
inner_step_count = 3

#s_matrix, ground_truth = SSBM(N,K)
#print(ground_truth)
#laplacian_matrix = data_generator(s_matrix, noise, sparsity)
G = nx.karate_club_graph()
gt_membership = [G.nodes[v]['club'] for v in G.nodes()]
gt_number = []
for i in gt_membership:
    if i == "Mr. Hi":
        gt_number.append(1)
    elif i =="Officer":
        gt_number.append(0)    
#gt_number = np.array(gt_number)

adj_mat = nx.convert_matrix.to_numpy_matrix(G)
laplacian_matrix = adjacancy_to_laplacian(adj_mat)
#V_output = mbo_modularity_own(N, K, m, dt, laplacian_matrix, tol, inner_step_count)
#ARI = adjusted_rand_score(V_output,gt_number)
#print('V_output: ', V_output)
#print('gt_number: ', gt_number)
#print(ARI)

## New version

import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigs, eigsh

from graph_mbo.utils import apply_threshold, get_fidelity_term, get_initial_state




"""
    Run the MBO scheme on a graph.
    Parameters
    ----------
    adj_matrix : np.array
        The adjacency matrix of the graph.
    normalized : bool
        Use the normalized graph Laplacian.
    signless : bool
        Use the signless graph Laplacian to find eigenvalues if normalized
    pseudospectral : bool
        Use the pseudospectral solver. If false, use CG or LU.
    m : int
        Number of eigenvalues to use for pseudospectral
    num_communities : int
        Number of communities
    target_size : list
        List of desired community sizes when using auction MBO
    thresh_type : str
        Type of thresholding to use. "max" takes the max across communities,
        "auction" does auction MBO
    dt : float
        Time step between thresholds for the MBO scheme
    min_dt : float
        Minimum time step for MBO convergence
    max_iter : int
        Maximum number of iterations
    n_inner : int
        Number of iterations for the MBO diffusion loop
    modularity : bool
        Add in the modularity minimization term
    """

def mbo_modularity(num_communities, m, dt, adj_matrix, tol ,inner_step_count, normalized=True,symmetric=True,
                       pseudospectral=True, target_size=None, signless=None, fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="fidelity", thresh_type="max", modularity=False): # inner stepcount is actually important! and can't be set to 1...
    
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=1)).flatten()
    num_nodes = len(degree)
    adj_positive = np.where(adj_matrix > 0, adj_matrix, 0)   # A_{ij}^+
    adj_negative = -np.where(adj_matrix < 0, adj_matrix, 0)   # A_{ij}^-

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    resolution = 1
    total_degree = np.sum(A_absolute_matrix)
    for i in range(len(degree)):
        for j in range(len(degree)):
            null_model[i][j] = resolution * ((degree[i] * degree[j]) / total_degree)

    if num_nodes < num_communities:
        print("Wrong input, N should larger or equal than K.")
    
    else:
        
        m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

        if target_size is None:
            target_size = [num_nodes // num_communities for i in range(num_communities)]
            target_size[-1] = num_nodes - sum(target_size[:-1])

        #graph_laplacian, degree = sp.sparse.csgraph.laplacian(A_absolute_matrix, return_diag=True)
        degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
        graph_laplacian = degree_diag - adj_matrix
        if symmetric:
            degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
            graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_{sym}
            # degree = np.ones(num_nodes)
            # degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
        elif normalized:
            degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
            graph_laplacian = degree_inv @ graph_laplacian
        

        if pseudospectral:
            if signless:
                if normalized:
                    pass
                else:
                    graph_laplacian = 2 * degree_diag - graph_laplacian   # Q=D+A=2D-L (unsigned)
            if normalized:
                D, V = eigs(
                    graph_laplacian,
                    k=m,
                    v0=np.ones((graph_laplacian.shape[0], 1)),
                    which="LR" if signless else "SR",
                )
            else:
                D, V = eigsh(
                    graph_laplacian,
                    k=m,
                    v0=np.ones((graph_laplacian.shape[0], 1)),
                    which="LA" if signless else "SA",
                )
            if signless:
                D = 2 * np.ones((m,)) - D  # Change D to be eigenvalues of graph Laplacian
            if normalized:
                # rescale eigenvectors to normalized space and orthogonalize
                for i in range(len(D)):
                    V[:, i] /= np.sqrt(V[:, i].transpose() @ degree_diag @ V[:, i])
        
        last_dt = 0

        if fidelity_type == "spectral":
            fidelity_D, fidelity_V = eigsh(
                graph_laplacian,
                k=num_communities + 1,
                v0=np.ones((graph_laplacian.shape[0], 1)),
                which="SA",
            )
            fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
            fidelity_D = fidelity_D[1:]
            # apply_threshold(fidelity_V, target_size, "max")
            # return fidelity_V
        else:
            fidelity_V = None


        # Initialize parameters
        u = get_initial_state(
            num_nodes,
            num_communities,
            target_size,
            type=initial_state_type,
            fidelity_type=fidelity_type,
            fidelity_V=fidelity_V,)


        last_last_index = u == 1
        last_index = u == 1
        last_dt = 0
        
        
        # Perform MBO scheme

        for n in range(max_iter):
            dti = dt / (2 * inner_step_count)

            if pseudospectral:

                if normalized:
                    a = V.transpose() @ (degree_inv @ u)  # Project into Hilbert space
                else:
                    a = V.transpose() @ u
                d = np.zeros((m, num_communities))
                demon = sp.sparse.spdiags([1 / (1 + dti * D)], [0], m, m)
                #demon = sp.sparse.spdiags([np.exp(-D*dt)],[0],m,m)
            else:
                if last_dt != dt:
                    lu, piv = lu_factor(sp.sparse.eye(num_nodes) + dti * graph_laplacian)

            
            for j in range(inner_step_count):
                
                # Solve system (apply CG or pseudospectral)
                if pseudospectral:
                    a = demon @ (a + fidelity_coeff * dti * d)
                    u = V @ a  # Project back into normal space
                    fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                    
                    # Project fidelity term into Hilbert space
                    if normalized:
                        d = V.transpose() @ (degree_inv @ fidelity_term)
                    else:
                        d = V.transpose() @ fidelity_term
                else:
                    fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                    u += fidelity_coeff * dti * fidelity_term
                   
                    if modularity:
                        # Add term for modularity
                        mean_f = np.dot(degree.reshape(1, len(degree)), u) / np.sum(degree)
                        # print("A")
                        # print(u)
                        # print(degree)
                        # print(mean_f)
                        # print(np.mean(u, axis=0))
                        # print(u[0,:])
                        # print("sum", np.sum(u, axis=0))
                        # print(degree[0])
                        # print((2 * dti * degree_diag @ (u - np.ones((u.shape[0], 1)) @ mean_f))[0,:])
                        # x = input()
                        # if x == "X":
                        #     raise Exception()
                        # @ (np.eye(u.shape[0]) - degree_diag / np.sum(degree))
                        u += 2 * dti * degree_diag @ (u - mean_f)

                    for i in range(num_communities):            
                        u[:, i] = lu_solve((lu, piv), u[:, i])

                j = j + 1
                

            # Apply thresholding 
            apply_threshold(u, target_size, thresh_type)

            # Stopping criterion 
            # Check that the index is changing and stop if time step becomes too small
            index = u == 1
            last_dt = dt

            norm_deviation = sp.linalg.norm(last_index ^ index) / sp.linalg.norm(index)
            if norm_deviation < tol :
                if dt < tol:
                    break
                else:
                    dt *= 0.5
            elif np.sum(last_last_index ^ index) == 0:
                # Going back and forth
                dt *= 0.5
            last_last_index = last_index
            last_index = index
            
            n = n+1

        if dt >= tol:
            print("MBO failed to converge")
        return u


# generate a random graph with community structure by the signed stochastic block model 
def SSBM_own(N, K):
    if N%K != 0:
        print("Wrong Input")

    else:
        s_matrix = -np.ones((N,N))
        cluster_size = N/K
        clusterlist = []
        for cs in range(K):
            clusterlist.append(int(cs*cluster_size))
        clusterlist.append(int(N))
        clusterlist.sort()
        #print(clusterlist)

        accmulate_size = []
        for quantity in range(len(clusterlist)-1):
            accmulate_size.append(clusterlist[quantity+1]-clusterlist[quantity])
        #print(accmulate_size)

        for interval in range(len(clusterlist)):
            for i in range(clusterlist[interval-1], clusterlist[interval]):
                for j in range(clusterlist[interval-1], clusterlist[interval]):
                    s_matrix[i][j] = 1
        #print(s_matrix)

        ground_truth = []
        for gt in range(len(accmulate_size)):
            ground_truth.extend([gt for y in range(accmulate_size[gt])])
        #print(ground_truth)

        ground_truth_v2 = []
        for gt in range(len(accmulate_size)):
            ground_truth_v2.extend([accmulate_size[gt] for y in range(accmulate_size[gt])])
        #print(ground_truth_v2)

    return s_matrix, ground_truth



def data_generator(s_matrix, noise, sparsity):
    # generate adjacancy matrix from s_matrix
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
    return A_matrix
