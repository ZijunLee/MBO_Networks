from joblib import PrintTime
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigs, eigsh
from random import randrange
import random
import numba as nb
from torch import sign

from graph_mbo.utils import apply_threshold, get_fidelity_term, get_initial_state,labels_to_vector,to_standard_labels,_diffusion_step_eig,_mbo_forward_step_multiclass,get_initial_state_1,ProjectToSimplex




"""
    Run the MBO scheme on a graph.
    Parameters
    ----------
    adj_matrix : np.array
        The adjacency matrix of the graph.
    normalized : bool
        Use the normalized graph Laplacian / normalized signless laplacian
    m : int
        Number of eigenvalues to use for pseudospectral
    num_communities : int
        Number of communities
    eta : int
        parameter in construct L_{mix} = L_F + eta * Q_H
    tol : scalar, 
        stopping criterion for iteration
    target_size : list
        List of desired community sizes when using auction MBO
    max_iter : int
        Maximum number of iterations
    inner_step : int
        Number of iterations for the MBO diffusion loop
    """


def mbo_modularity_1(num_communities, m, adj_matrix, tol, eta,eps=1,
                    target_size=None, max_iter=10000): 
    
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print('max degree: ',np.max(degree))
    num_nodes = len(degree)
        
    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute unsigned laplacian L_F
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix    
    #print('graph laplacian shape: ',graph_laplacian.shape)

    # compute symmetric normalized laplacian
    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    for i in range(len(degree)):
        for j in range(len(degree)):
            null_model[i][j] = (degree[i] * degree[j]) / total_degree
    #print('null model shape: ', null_model.shape)
        
    # compute signless laplacian Q_H
    degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    num_nodes_null_model = len(degree_null_model)
    degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_H = D + H(null model)
    
    # Construct L_{mix} = symmetric normalized L_F + unnormalized Q_H
    laplacian_mix = graph_laplacian + eta * signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P
    #print('L_{mix} shape: ',laplacian_mix.shape)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    # initialized u 
    u = get_initial_state_1(num_nodes, num_communities,target_size)

    #Random initial labeling
    #u = np.random.rand(num_nodes,num_communities)
    #u = ProjectToSimplex(u)


    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    #Time step selection
    dtlow = 0.15/((eta+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        # Diffusion
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        u_half = V_sign @ (demon @ u_old)  
                
        # Apply thresholding 
        u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)             
                    
        # Stop criterion
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n + 1
        #print(n)

    return u_new, n




def mbo_modularity_1_normalized_lf(num_communities, m, adj_matrix, tol, eta,eps=1,
                       target_size=None, max_iter=10000): 
    
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print('max degree: ',np.max(degree))
    num_nodes = len(degree)
        
    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute unsigned laplacian
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
    #print('graph laplacian shape: ',graph_laplacian.shape)

    # compute symmetric normalized laplacian
    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    for i in range(len(degree)):
        for j in range(len(degree)):
            null_model[i][j] = (degree[i] * degree[j]) / total_degree
    
    #print('null model: ',null_model)
    
    degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    num_nodes_null_model = len(degree_null_model)
    degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    #print('signless numm model: ',signless_laplacian_null_model)
    
    # Construct L_{mix} = symmetric normalized L_F + unnormalized Q_H
    laplacian_mix = nor_graph_laplacian +  eta * signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P

    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    # initialized u 
    u = get_initial_state_1(num_nodes, num_communities,target_size)

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    #Time step selection
    dtlow = 0.15/((eta+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        # Diffusion 
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        u_half = V_sign @ (demon @ u_old)  

        # Apply thresholding 
        u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                                 
        # Stop criterion
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n+1
        #print(n)

    return u_new, n



def mbo_modularity_1_normalized_Qh(num_communities, m, adj_matrix, tol, eta,eps=1,
                       target_size=None, max_iter=10000):
    
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print('max degree: ',np.max(degree))
    num_nodes = len(degree)
        
    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute standard laplacian
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
    #print('graph laplacian shape: ',graph_laplacian.shape)

    # compute symmetric normalized laplacian
    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    for i in range(len(degree)):
        for j in range(len(degree)):
            null_model[i][j] = (degree[i] * degree[j]) / total_degree
    
    #print('null model: ',null_model)
    
    # compute signless symmetric normalized laplacian Q_h
    degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    num_nodes_null_model = len(degree_null_model)
    degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    signless_degree_inv = sp.sparse.spdiags([1.0 / degree_null_model], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_signless_laplacian = np.sqrt(signless_degree_inv) @ signless_laplacian_null_model @ np.sqrt(signless_degree_inv)    # obtain L_A_{sym}

    # Construct L_{mix} = unnormalized L_F + normalized Q_H
    laplacian_mix = graph_laplacian +  eta * nor_signless_laplacian  # L_{mix} = L_A + Q_P_{sym}
    #print('L_{mix}: ',laplacian_mix)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    # initialized u 
    u = get_initial_state_1(num_nodes, num_communities,target_size)


    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    #Time step selection
    dtlow = 0.15/((eta+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        # Diffusion
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        u_half = V_sign @ (demon @ u_old)  

        # Apply thresholding 
        u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                             
        # Stop criterion
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n + 1
        #print(n)

    return u_new, n



def mbo_modularity_1_normalized_Lf_Qh(num_communities, m, adj_matrix, tol, eta,eps=1,
                       target_size=None, max_iter=10000): # inner stepcount is actually important! and can't be set to 1...
    
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print('max degree: ',np.max(degree))
    num_nodes = len(degree)
        
    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute standard laplacian
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
    #print('graph laplacian shape: ',graph_laplacian.shape)

    # compute symmetric normalized laplacian
    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    #resolution = 1
    total_degree = np.sum(adj_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    for i in range(len(degree)):
        for j in range(len(degree)):
            null_model[i][j] = (degree[i] * degree[j]) / total_degree
    
    #print('null model: ',null_model)
    
    # compute signless symmetric normalized laplacian Q_h
    degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    num_nodes_null_model = len(degree_null_model)
    degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    signless_degree_inv = sp.sparse.spdiags([1.0 / degree_null_model], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_signless_laplacian = np.sqrt(signless_degree_inv) @ signless_laplacian_null_model @ np.sqrt(signless_degree_inv)    # obtain L_A_{sym}

    # Construct L_{mix} = symmetric normalized L_F + normalized Q_H
    laplacian_mix = nor_graph_laplacian +  eta * nor_signless_laplacian  # L_{mix} = L_A_{sym} + Q_P_{sym}
    #print('L_{mix}: ',laplacian_mix)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    # initialized u 
    u = get_initial_state_1(num_nodes, num_communities,target_size)

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
    
    #Time step selection
    dtlow = 0.15/((eta+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
        
    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        # Diffusion
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        u_half = V_sign @ (demon @ u_old)  

        # Apply thresholding 
        u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                    
                    
        # Stop criterion
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n + 1
        #print(n)

    return u_new, n