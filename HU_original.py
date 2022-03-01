from joblib import PrintTime
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigs, eigsh
from random import randrange
import random
#import numba as nb
from torch import sign

from graph_mbo.utils import apply_threshold, get_fidelity_term, get_initial_state,labels_to_vector,to_standard_labels,_diffusion_step_eig,_mbo_forward_step_multiclass,get_initial_state_1,ProjectToSimplex




def mbo_modularity_hu_original(num_communities, m, dt, adj_matrix, tol ,inner_step_count, 
                               target_size=None, max_iter=10000, thresh_type="max", modularity=False): # inner stepcount is actually important! and can't be set to 1...
    
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    num_nodes = len(degree)

    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    #graph_laplacian, degree = sp.sparse.csgraph.laplacian(A_absolute_matrix, return_diag=True)
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix

    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
    graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_{sym}
    # degree = np.ones(num_nodes)
    # degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)

    D, V = eigsh(
        graph_laplacian,
        k=m,
        v0=np.ones((graph_laplacian.shape[0], 1)),
        which="SA")


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)

    last_last_index = u == 1
    last_index = u == 1
    last_dt = 0
    #stop_criterion = 10
    #u_new = u.copy()        
    # Perform MBO scheme

    for n in range(max_iter):
        #u_old = u_new.copy()
        dti = dt / inner_step_count

        demon = sp.sparse.spdiags([1 / (1 + dti * D)], [0], m, m) @ V.transpose()
        #demon = sp.sparse.spdiags([np.exp(-D*dt)],[0],m,m)
        
        for j in range(inner_step_count):
            
            u = V @ (demon @ u)
                
            if modularity:
                # Add term for modularity
                mean_f = np.dot(degree.reshape(1, len(degree)), u) / np.sum(degree)
                u += 2 * dti * degree_diag @ (u - mean_f)

            j = j + 1
            

        # Apply thresholding 
        u = apply_threshold(u, target_size, thresh_type)

        # Check that the index is changing and stop if time step becomes too small
        index = u == 1

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

