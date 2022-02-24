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
    unsign : bool
        Use Newman-Girvan null model as P_ij in unsigned network
    """

def adj_to_laplacian_signless_laplacian(adj_matrix,num_communities,m,gamma, target_size=None):
        
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=1)).flatten()
    dergee_di_null = np.sum(A_absolute_matrix, axis=1)
    #print('max degree: ',degree.shape)
    #print('degree d_i type: ', dergee_di_null.shape)
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

    # compute Random walk normalized Laplacian
    random_walk_nor_lap =  degree_inv @ graph_laplacian

    
    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    #for i in range(len(degree)):
    #    for j in range(len(degree)):
    #        null_model[i][j] = (degree[i] * degree[j]) / total_degree

    #null_model = (np.dot(np.transpose(degree), degree))/total_degree
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree
    null_model_eta = gamma * null_model

    #print('null model shape: ', null_model_eta.shape)
    
    degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    num_nodes_null_model = len(degree_null_model)
    degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    signless_degree_inv = sp.sparse.spdiags([1.0 / degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   # obtain D^{-1}
    #print('D^{-1}: ', degree_inv.shape)
    nor_signless_laplacian = np.sqrt(signless_degree_inv) @ signless_laplacian_null_model @ np.sqrt(signless_degree_inv)
    rw_signless_lapclacian =  signless_degree_inv @ signless_laplacian_null_model

    return num_nodes,m, degree, target_size,null_model_eta,graph_laplacian, nor_graph_laplacian,random_walk_nor_lap, signless_laplacian_null_model, nor_signless_laplacian, rw_signless_lapclacian
    

#@nb.jit()
def mbo_modularity_1(num_nodes,num_communities, m,degree, graph_laplacian,signless_laplacian_null_model, tol, target_size,
                    gamma, eps=1, fidelity_type="karate", max_iter=10000,
                    fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    laplacian_mix = graph_laplacian + signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P
    #print('L_{mix} shape: ',laplacian_mix.shape)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        laplacian_mix,
    #        k=num_communities + 1,
    #        v0=np.ones((laplacian_mix.shape[0], 1)),
    #        which="SA",
    #    )
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)

    # Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)
        
    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
 
    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        
        #for j in range(inner_step_count):
            
            # Solve system (apply CG or pseudospectral)

        u_half = V_sign @ (demon @ u_old)  # Project back into normal space

        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        
        #Ui_diff = []
        #Ui_max = []
        #for i in range(num_nodes):    
        #    Ui_diff.append((np.linalg.norm(u_new[i,:] - u_old[i,:]))**2)
        #    Ui_max.append((np.linalg.norm(u_new[i,:]))**2)
            
        #max_diff = max(Ui_diff)
        #max_new = max(Ui_max)
        #stop_criterion = max_diff/max_new

        n = n+1
        #print(n)

    return u_new, n


# generate a random graph with community structure by the signed stochastic block model 
def SSBM_own(N, K):
    if N%K != 0:
        print("Wrong Input")

    else:
        #s_matrix = -np.ones((N,N))
        s_matrix = np.zeros((N,N))
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



#@nb.jit(nopython=True)
def mbo_modularity_2(num_communities, m, adj_matrix, tol,gamma,eps=1,
                       target_size=None, fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=1)).flatten()
    dergee_di_null = np.sum(adj_matrix, axis=1)
    num_nodes = len(degree)
    #print(num_nodes)

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(A_absolute_matrix)
    #print('total degree: ', total_degree)
    #print('length of degree: ', len(degree))

    #for i in range(len(degree)):
    #    for j in range(len(degree)):
    #        null_model[i][j] = (degree[i] * degree[j]) / total_degree

    #null_model = (np.dot(np.transpose(degree), degree))/total_degree
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree
    null_model_eta = gamma * null_model

    #degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    #num_nodes_null_model = len(degree_null_model)
    #degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    #signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)

    # method 2: B = W - P  (signed), compute signed laplacian
    mix_matrix = adj_matrix - null_model_eta   # B = W - P (signed)
    mix_matrix_absolute = np.abs(mix_matrix)
    degree_mix_mat = np.array(np.sum(mix_matrix_absolute, axis=1)).flatten()
    num_nodes_mix_mat = len(degree_mix_mat)
    #print(num_nodes_mix_mat)
    degree_diag_mix_mat = sp.sparse.spdiags([degree_mix_mat], [0], num_nodes_mix_mat, num_nodes_mix_mat)  # Dbar
#    laplacian_mix_mat =  degree_diag_mix_mat - mix_matrix  # L_B = Dbar - B

#    laplacian_mix = graph_laplacian_positive + signless_graph_laplacian  # L_{mix} = L_B^+ + Q_B^- = Lbar
#    laplacian_mix_mat = 0.5*laplacian_mix_mat

    # compute signed laplacian
        #graph_laplacian, degree = sp.sparse.csgraph.laplacian(A_absolute_matrix, return_diag=True)
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)  # Dbar
    #sign_graph_laplacian = degree_diag - adj_matrix    # L_A = Dbar - A

    # B_{ij}^+
    mix_mat_positive = np.where(mix_matrix > 0, mix_matrix, 0)   # B_{ij}^+
    degree_mix_mat_positive = np.array(np.sum(mix_mat_positive, axis=1)).flatten()
    num_nodes_positive = len(degree_mix_mat_positive)
    degree_diag_positive = sp.sparse.spdiags([degree_mix_mat_positive], [0], num_nodes_positive, num_nodes_positive)  # D^+
    graph_laplacian_positive = degree_diag_positive - mix_mat_positive  # L_B^+ = D^+ - B^+ 

    # B_{ij}^-
    mix_mat_negative = -np.where(mix_matrix < 0, mix_matrix, 0)   # B_{ij}^-
    degree_mix_mat_negative = np.array(np.sum(mix_mat_negative, axis=1)).flatten()
    num_nodes_negative = len(degree_mix_mat_negative)
    degree_diag_negative = sp.sparse.spdiags([degree_mix_mat_negative], [0], num_nodes_negative, num_nodes_negative)  # D^-
    signless_graph_laplacian = degree_diag_negative + mix_mat_negative  # Q_B^- = D^- + B^-

    # compute symmetric normalized laplacian
    laplacian_mix_mat = graph_laplacian_positive + signless_graph_laplacian  # L_{mix} = L_B^+ + Q_B^- = Lbar
#    laplacian_mix_mat = 0.5 * laplacian_mix_mat

    degree_inv = sp.sparse.spdiags([1.0 / degree_mix_mat], [0], num_nodes_mix_mat, num_nodes_mix_mat)   # obtain Dbar^{-1}
    sym_graph_laplacian = np.sqrt(degree_inv) @ laplacian_mix_mat @ np.sqrt(degree_inv)    # obtain L_{sym}
#    sign_graph_laplacian = 0.5 * sign_graph_laplacian

        
    m = min(num_nodes_mix_mat - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes_mix_mat // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes_mix_mat - sum(target_size[:-1])

        
    # eigendecomposition signed symmetric normalized laplacian
    D_sign, V_sign = eigsh(
        sym_graph_laplacian,
        k=m,
        v0=np.ones((sym_graph_laplacian.shape[0], 1)),
        which= "SA",)


    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        sym_graph_laplacian,
    #        k=num_communities + 1,
    #        v0=np.ones((sym_graph_laplacian.shape[0], 1)),
    #        which="SA",)
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes_mix_mat,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)
    
    u = get_initial_state_1(num_nodes, num_communities, target_size)


    # Time step selection
    dtlow = 0.15/((gamma+1) * np.max(degree_mix_mat))
    dthigh = np.log(np.linalg.norm(u) / eps) / D_sign[0]
    dti = np.sqrt(dtlow * dthigh)
    
    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
    #for n in range(max_iter):
    while (n < max_iter) and (stop_criterion > tol):

        u_old = u_new.copy()
    
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        
    #    for j in range(inner_step_count):
            
            # Solve system (apply CG or pseudospectral)
        #    if pseudospectral:
            #    a = demon @ (a + fidelity_coeff * dti * d)
        #a = demon @ a
        u_half = V_sign @ (demon @ u_old)  # Project back into normal space      

        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)

                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        
        #Ui_diff = []
        #Ui_max = []
        #for i in range(num_nodes_mix_mat):    
        #    Ui_diff.append((np.linalg.norm(u_new[i,:] - u_old[i,:]))**2)
        #    Ui_max.append((np.linalg.norm(u_new[i,:]))**2)
            
        #max_diff = max(Ui_diff)
        #max_new = max(Ui_max)
        #stop_criterion = max_diff/max_new

        n = n+1

    return u_new, n



## given eigenvalues and eigenvectors
def mbo_modularity_given_eig(eigval,eigvec, u_init,k_weights,tol=0.5, gamma=0.5,eps=1,max_iter=10000): 
    
    degree = np.array(k_weights).flatten
    Neig = eigvec.shape[1]
    #degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print(np.max(degree))
    #num_nodes = len(degree)
        
    #m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    #if target_size is None:
    #    target_size = [num_nodes // num_communities for i in range(num_communities)]
    #    target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute unsigned laplacian
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    #graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
    # compute symmetric normalized laplacian
    #degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    #nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    ## Construct Newman--Girvan null model
    #null_model = np.zeros((len(degree), len(degree)))
    #resolution = 1
    #total_degree = np.sum(adj_matrix)
    #for i in range(len(degree)):
    #    for j in range(len(degree)):
    #        null_model[i][j] = gamma * ((degree[i] * degree[j]) / total_degree)

    #degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    #num_nodes_null_model = len(degree_null_model)
    #degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    #signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    
    #laplacian_mix = 1 * (graph_laplacian + signless_laplacian_null_model)  # L_{mix} = L_A_{sym} + Q_P
    
    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which= "SA",)


    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        laplacian_mix,
    #        k=num_communities + 1,
    #        v0=np.ones((laplacian_mix.shape[0], 1)),
    #        which="SA",
    #    )
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)


    # convert u_init to standard multiclass form for binary tags
    if (len(u_init.shape)== 1) or (u_init.shape[1] == 1): 
        u_init = labels_to_vector(to_standard_labels(u_init))
    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u_init.copy()

    #Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u_init)/eps)/eigval[0]
    dti = np.sqrt(dtlow*dthigh)
        
    while (n < max_iter) and (stop_criterion > tol):

        u_old = u_new.copy()

        #dti = dt / (2 * inner_step_count)

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        demon = sp.sparse.spdiags([np.exp(- 0.5 * eigval * dti)],[0],Neig,Neig) @ eigvec.transpose()
        #a = demon @ a
        #demon = sp.sparse.spdiags([1.0 / (1.0 + dti * D_sign)], [0], m, m) @ V_sign.transpose()
        #    P = sp.sparse.spdiags([np.exp(-D_sign*dti)],[0],m,m) @ V_sign.T
        
        #for j in range(inner_step_count):
            
            # Solve system (apply CG or pseudospectral)

        u_half = eigvec @ (demon @ u_old)  # Project back into normal space
            #    u = V_sign @ (P @ u)
            #fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                
                # Project fidelity term into Hilbert space
                #if normalized:
                #    d = V_sign.transpose() @ (degree_inv @ fidelity_term)
                #    d = V_sign.transpose() @ (eigenvalue_mat @ fidelity_term)
                #else:
                #d = V_sign.transpose() @ fidelity_term
                
        # Apply thresholding 
        #u_new = apply_threshold(u_half, target_size, thresh_type)
        u_new = _mbo_forward_step_multiclass(u_half)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        
        #Ui_diff = []
        #Ui_max = []
        #for i in range(num_nodes):    
        #    Ui_diff.append((np.linalg.norm(u_new[i,:] - u_old[i,:]))**2)
        #    Ui_max.append((np.linalg.norm(u_new[i,:]))**2)
            
        #max_diff = max(Ui_diff)
        #max_new = max(Ui_max)
        #stop_criterion = max_diff/max_new

        n = n+1

    return u_new, n



def mbo_modularity_inner_step(num_nodes, num_communities, m, graph_laplacian, signless_laplacian_null_model,dt, tol,target_size,inner_step_count,
                       fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    
    laplacian_mix = graph_laplacian + signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P
    #print('L_{mix} shape: ',laplacian_mix.shape)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)


    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
    #for n in range(max_iter):
    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        dti = dt / (2 * inner_step_count)

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        #demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        #print('demon shape: ', demon.shape)
        #a = demon @ a
        demon = sp.sparse.spdiags([1.0 / (1.0 + dti * D_sign)], [0], m, m) @ V_sign.transpose()
        #    P = sp.sparse.spdiags([np.exp(-D_sign*dti)],[0],m,m) @ V_sign.T
        
        for j in range(inner_step_count):
            
            # Solve system (apply CG or pseudospectral)
            u_half = V_sign @ (demon @ u_old)  # Project back into normal space

                
        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new shape: ',u_new,shape)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n+1
        #print(n)

    return u_new, n



def mbo_modularity_1_normalized_lf(num_nodes,num_communities, m, degree,nor_graph_laplacian,signless_laplacian_null_model, 
                       tol,target_size, gamma,eps=1, fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    laplacian_mix = nor_graph_laplacian + signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P
    #print('L_{mix}: ',laplacian_mix)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        laplacian_mix,
    #        k=num_communities + 1,
    #        v0=np.ones((laplacian_mix.shape[0], 1)),
    #        which="SA",
    #    )
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V)

    u = get_initial_state_1(num_nodes, num_communities, target_size)

    #Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    while (n < max_iter) and (stop_criterion > tol):
        
        u_old = u_new.copy()

        #dti = dt / (2 * inner_step_count)

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        
        #for j in range(inner_step_count):
            
            # Solve system (apply CG or pseudospectral)

        u_half = V_sign @ (demon @ u_old)  # Project back into normal space
        #print('u_half: ',u_half)
            #    u = V_sign @ (P @ u)

        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
       # u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n+1
        #print(n)

    return u_new, n



def mbo_modularity_1_normalized_Qh(num_nodes,num_communities, m,degree, graph_laplacian,nor_signless_laplacian,
                        tol,target_size, gamma,eps=1, fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    
    laplacian_mix = graph_laplacian + nor_signless_laplacian  # L_{mix} = L_A + Q_P_{sym}
    #print('L_{mix}: ',laplacian_mix)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        laplacian_mix,
    #        k=num_communities + 1,
    #        v0=np.ones((laplacian_mix.shape[0], 1)),
    #        which="SA",
    #    )
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)

    # Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    while (n < max_iter) and (stop_criterion > tol):

        u_old = u_new.copy()

        #dti = dt / (2 * inner_step_count)

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        #print('demon shape: ', demon.shape)
        #a = demon @ a
        #demon = sp.sparse.spdiags([1.0 / (1.0 + dti * D_sign)], [0], m, m) @ V_sign.transpose()
        #    P = sp.sparse.spdiags([np.exp(-D_sign*dti)],[0],m,m) @ V_sign.T

        u_half = V_sign @ (demon @ u_old)  # Project back into normal space
        #print('u_half: ',u_half)
            #    u = V_sign @ (P @ u)

        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n + 1
        #print(n)

    return u_new, n



def mbo_modularity_1_normalized_Lf_Qh(num_nodes,num_communities, m,degree, nor_graph_laplacian,nor_signless_laplacian, 
                       tol, target_size,gamma, eps=1, fidelity_type="karate", max_iter=10000,
                       fidelity_coeff=10, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    laplacian_mix = nor_graph_laplacian + nor_signless_laplacian  # L_{mix} = L_A_{sym} + Q_P_{sym}
    #print('L_{mix}: ',laplacian_mix)
    
    # compute eigenvalues and eigenvectors
    D_sign, V_sign = eigsh(
        laplacian_mix,
        k=m,
        v0=np.ones((laplacian_mix.shape[0], 1)),
        which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    #if fidelity_type == "spectral":
    #    fidelity_D, fidelity_V = eigsh(
    #        laplacian_mix,
    #        k=num_communities + 1,
    #        v0=np.ones((laplacian_mix.shape[0], 1)),
    #        which="SA",
    #    )
    #    fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
    #    fidelity_D = fidelity_D[1:]
    #    # apply_threshold(fidelity_V, target_size, "max")
    #    # return fidelity_V
    #else:
    #    fidelity_V = None


    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)

    # Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()

    while (n < max_iter) and (stop_criterion > tol):
        
        u_old = u_new.copy()

        #dti = dt / (2 * inner_step_count)

        #if pseudospectral:

        #a = V_sign.transpose() @ u_old

        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()

        u_half = V_sign @ (demon @ u_old)  # Project back into normal space
        #print('u_half: ',u_half)
            #    u = V_sign @ (P @ u)

        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)
        #print('u_new: ',u_new)
                    
                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        
        #Ui_diff = []
        #Ui_max = []
        #for i in range(num_nodes):    
        #    Ui_diff.append((np.linalg.norm(u_new[i,:] - u_old[i,:]))**2)
        #    Ui_max.append((np.linalg.norm(u_new[i,:]))**2)
            
        #max_diff = max(Ui_diff)
        #max_new = max(Ui_max)
        #stop_criterion = max_diff/max_new

        n = n + 1
        #print(n)

    return u_new, n