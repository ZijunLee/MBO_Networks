#from joblib import PrintTime
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, eigsh, svds
#from random import randrange
#import random
#from torch import sign
import time
#import quimb

from graph_mbo.utils import apply_threshold,_diffusion_step_eig,_mbo_forward_step_multiclass,get_initial_state_1
#from graph_cut.util.nystrom import nystrom_extension
#from slec4py_test2 import eigs_slepc



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


def construct_null_model(adj_matrix):
        
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=-1)).flatten()

    del A_absolute_matrix

    dergee_di_null = np.expand_dims(degree, axis=-1)
    #print('max degree: ',degree.shape)
    #print('degree d_i type: ', dergee_di_null.shape)

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    total_degree_int = total_degree.astype(int)
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int

    return null_model
    


def adj_to_laplacian_signless_laplacian(adj_matrix, num_communities, m, target_size=None):
        
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=-1)).flatten()
    num_nodes = len(degree)

    start_time_construct_null_model = time.time()
    dergee_di_null = np.expand_dims(degree, axis=-1)
    #print('dergee_di_null: ', type(dergee_di_null))
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    total_degree_int = total_degree.astype(int)
    #print('total_degree_int: ', type(total_degree_int))
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int
    time_null_model = time.time() - start_time_construct_null_model
    print("construct null model:-- %.3f seconds --" % (time_null_model))

    del A_absolute_matrix, dergee_di_null
        
    m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes - sum(target_size[:-1])

    start_time_construct_lap_signless = time.time()
    # compute unnormalized laplacian
    degree_diag = np.diag(degree)
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
 
    del degree_diag, adj_matrix

    # compute symmetric normalized laplacian L_F
    degree_inv = np.diag((1 / degree))
    #degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}
    
    # compute random walk normalized Laplacian
    random_walk_nor_lap =  degree_inv @ graph_laplacian

    del degree_inv

    # compute unnormalized laplacian
    degree_null_model = np.array(np.sum(null_model, axis=-1)).flatten()
    #degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)
    degree_diag_null_model = np.diag(degree_null_model)   
    signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
    #signless_degree_inv = sp.sparse.spdiags([1.0 / degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   # obtain D^{-1}
    
    del degree_diag_null_model, null_model

    # compute symmetric normalized signless laplacian Q_H
    signless_degree_inv = np.diag((1.0/degree_null_model))
    
    del degree_null_model

    nor_signless_laplacian = np.sqrt(signless_degree_inv) @ signless_laplacian_null_model @ np.sqrt(signless_degree_inv)
    rw_signless_lapclacian =  signless_degree_inv @ signless_laplacian_null_model

    time_laplacian = time.time() - start_time_construct_lap_signless
    print("construct laplacian & signless laplacian:-- %.3f seconds --" % (time_laplacian))

    del signless_degree_inv

    return num_nodes,m, degree, target_size,graph_laplacian, nor_graph_laplacian,random_walk_nor_lap, signless_laplacian_null_model, nor_signless_laplacian, rw_signless_lapclacian
    


def MMBO2_preliminary(adj_matrix, num_communities,m,gamma, target_size=None):
    
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=1)).flatten()
    dergee_di_null = np.expand_dims(degree, axis=-1)

    num_nodes = len(degree)
    #print(num_nodes)

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(A_absolute_matrix)
    total_degree_int = total_degree.astype(int)
    print('total_degree_int: ', type(total_degree_int))
    #print('length of degree: ', len(degree))

    #for i in range(len(degree)):
    #    for j in range(len(degree)):
    #        null_model[i][j] = (degree[i] * degree[j]) / total_degree

    #null_model = (np.dot(np.transpose(degree), degree))/total_degree
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int
    #null_model_eta = gamma * null_model

    #degree_null_model = np.array(np.sum(null_model, axis=1)).flatten()
    #num_nodes_null_model = len(degree_null_model)
    #degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   
    #signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)

    # method 2: B = W - P  (signed), compute signed laplacian
    mix_matrix = adj_matrix - null_model   # B = W - P (signed)
    mix_matrix_absolute = np.abs(mix_matrix)
    degree_mix_mat = np.array(np.sum(mix_matrix_absolute, axis=1)).flatten()
    num_nodes_mix_mat = len(degree_mix_mat)
    #print(num_nodes_mix_mat)
    #degree_diag_mix_mat = sp.sparse.spdiags([degree_mix_mat], [0], num_nodes_mix_mat, num_nodes_mix_mat)  # Dbar
    #laplacian_mix_mat =  degree_diag_mix_mat - mix_matrix  # L_B = Dbar - B

    #laplacian_mix = graph_laplacian_positive + signless_graph_laplacian  # L_{mix} = L_B^+ + Q_B^- = Lbar

    # compute signed laplacian
        #graph_laplacian, degree = sp.sparse.csgraph.laplacian(A_absolute_matrix, return_diag=True)
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)  # Dbar
    #sign_graph_laplacian = degree_diag - adj_matrix    # L_A = Dbar - A

    # B_{ij}^+
    mix_mat_positive = np.where(mix_matrix > 0, mix_matrix, 0)   # B_{ij}^+
    degree_mix_mat_positive = np.array(np.sum(mix_mat_positive, axis=1)).flatten()
    #num_nodes_positive = len(degree_mix_mat_positive)
    #degree_diag_positive = sp.sparse.spdiags([degree_mix_mat_positive], [0], num_nodes_positive, num_nodes_positive)  # D^+
    degree_diag_positive = np.diag(degree_mix_mat_positive)
    graph_laplacian_positive = degree_diag_positive - mix_mat_positive  # L_B^+ = D^+ - B^+ 
    
    # compute symmetric normalized laplacian & Random walk normalized Laplacian
    #degree_inv_positive = sp.sparse.spdiags([1.0 / degree_mix_mat_positive], [0], num_nodes_positive, num_nodes_positive)   # obtain D^{-1}
    degree_inv_positive = np.diag((1.0 /degree_mix_mat_positive))
    #print('D^{-1}: ', degree_inv.shape)
    sym_graph_laplacian_positive = np.sqrt(degree_inv_positive) @ graph_laplacian_positive @ np.sqrt(degree_inv_positive)    # obtain L_A_{sym}
    random_walk_nor_lap_positive =  degree_inv_positive @ graph_laplacian_positive

    # B_{ij}^-
    mix_mat_negative = -np.where(mix_matrix < 0, mix_matrix, 0)   # B_{ij}^-
    degree_mix_mat_negative = np.array(np.sum(mix_mat_negative, axis=1)).flatten()
    #num_nodes_negative = len(degree_mix_mat_negative)
    degree_diag_negative = np.diag(degree_mix_mat_negative)
    #degree_diag_negative = sp.sparse.spdiags([degree_mix_mat_negative], [0], num_nodes_negative, num_nodes_negative)  # D^-
    signless_graph_laplacian_neg = degree_diag_negative + mix_mat_negative  # Q_B^- = D^- + B^-
        
    # compute symmetric normalized laplacian & Random walk normalized Laplacian
    #degree_inv_negative = sp.sparse.spdiags([1.0 / degree_mix_mat_negative], [0], num_nodes_negative, num_nodes_negative)   # obtain D^{-1}
    degree_inv_negative = np.diag((1.0 / degree_mix_mat_negative))
    #print('D^{-1}: ', degree_inv.shape)
    sym_signless_laplacian_negative = np.sqrt(degree_inv_negative) @ signless_graph_laplacian_neg @ np.sqrt(degree_inv_negative)    # obtain L_A_{sym}
    random_walk_signless_lap_negative =  degree_inv_negative @ signless_graph_laplacian_neg

    m = min(num_nodes_mix_mat - 2, m)  # Number of eigenvalues to use for pseudospectral

    if target_size is None:
        target_size = [num_nodes_mix_mat // num_communities for i in range(num_communities)]
        target_size[-1] = num_nodes_mix_mat - sum(target_size[:-1])


    return num_nodes, m, target_size, graph_laplacian_positive, sym_graph_laplacian_positive, random_walk_nor_lap_positive, signless_graph_laplacian_neg, sym_signless_laplacian_negative, random_walk_signless_lap_negative




def mbo_modularity_1(num_nodes,num_communities, m, dt, u_init,laplacian_mix, eigval, eigvec, tol, target_size,
                    gamma, eps=1, max_iter=10000, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    print('Start with MMBO using the projection on the eigenvectors')
    
    #start_time_l_mix = time.time()
    #laplacian_mix = graph_lap + signless_lap
    #print("compute l_{mix}:-- %.3f seconds --" % (time.time() - start_time_l_mix))

    start_time_eigendecomposition = time.time()
    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    sigma=0,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which='SA')

    #eigenpair = eigs_slepc(laplacian_mix, m, which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
    #eigenpair = quimb.linalg.slepc_linalg.eigs_slepc(laplacian_mix, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))
    #print('EPSType is krylovschur')
    #D_sign = eigenpair[0]
    #V_sign = eigenpair[1]

    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)

    #start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    
    #start_time_timestep_selection = time.time()
    # Time step selection
    #dtlow = 0.15/((gamma+1)*np.max(degree))
    #dthigh = np.log(np.linalg.norm(u)/eps)/D_sign[0]
    #dti = np.sqrt(dtlow*dthigh)
    #print('dti: ',dti)
    #print("compute time step selection:-- %.3f seconds --" % (time.time() - start_time_timestep_selection))
    dti = dt 

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u_init.copy()
    
    start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):

        u_old = u_new.copy()

        # Diffusion step
        #start_time_diffusion = time.time()
        demon = sp.sparse.spdiags([np.exp(- 0.5 * eigval * dti)],[0],m,m) @ eigvec.transpose()
        # Solve system (apply CG or pseudospectral)
        u_half = eigvec @ (demon @ u_old)  # Project back into normal space
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding))            
                    
        # Stop criterion
        #start_time_stop_criterion = time.time()
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        
        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion))

        n = n + 1
    
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))
    
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



def mbo_modularity_2(num_nodes, num_communities, m,dti, tol, graph_laplacian_positive, signless_graph_laplacian,
                     target_size,eps=1, max_iter=10000, initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    # compute symmetric normalized laplacian
    laplacian_mix_mat = graph_laplacian_positive + signless_graph_laplacian  # L_{mix} = L_B^+ + Q_B^- = Lbar
        
    # eigendecomposition signed symmetric normalized laplacian
    D_sign, V_sign = eigsh(
        laplacian_mix_mat,
        k=m,
        v0=np.ones((laplacian_mix_mat.shape[0], 1)),
        which= "SA",)


    # Initialize parameters
    u = get_initial_state_1(num_nodes, num_communities, target_size)


    # Time step selection
    #dtlow = 0.15/((gamma+1) * np.max(degree_mix_mat))
    #dthigh = np.log(np.linalg.norm(u) / eps) / D_sign[0]
    #dti = np.sqrt(dtlow * dthigh)
    
    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
    #for n in range(max_iter):
    while (n < max_iter) and (stop_criterion > tol):

        u_old = u_new.copy()
    
        # Diffusion step
        demon = sp.sparse.spdiags([np.exp(- 0.5 * D_sign * dti)],[0],m,m) @ V_sign.transpose()
        # Solve system (apply CG or pseudospectral)
        u_half = V_sign @ (demon @ u_old)  # Project back into normal space      

        # Apply thresholding 
        #u_new = apply_threshold(u_half, target_size, thresh_type)
        u_new = _mbo_forward_step_multiclass(u_half)

                    
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)

        n = n+1

    return u_new, n



## given eigenvalues and eigenvectors
def mbo_modularity_given_eig(num_communities, num_nodes,degree,m,u_init, eigval,eigvec,dt, tol, 
                            inner_step_count, gamma=0.5,max_iter=10000, thresh_type="max"): 
    

    #m = len(eigval)
    #print('m: ',m)

    #eigval = eigval.reshape((m,))
    #print('eigenvalue shape: ', eigval.shape)

    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    #num_nodes = len(deg)
    #degree_diag = np.diag(degree)
    #degree_diag = sp.sparse.spdiags([deg], [0], num_nodes, num_nodes)
    #degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print(np.max(degree))
    #num_nodes = len(degree)
    
    #target_size = [num_nodes // num_communities for i in range(num_communities)]
    #target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which= "SA",)

    # Initialize parameters
    start_time_initialize = time.time()
    #u_init = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    
    # Perform MBO scheme
    #stop_criterion = 10
    #n = 0
    #u_new = u_init.copy() 
    
    #start_time_MBO_iteration = time.time()
    #while (n < max_iter) and (stop_criterion > tol):
        #u_old = u_new.copy()

        #start_time_diffusion = time.time()
        #demon = sp.sparse.spdiags([np.exp(- 0.5 * eigval * dt)],[0],m,m) @ eigvec.transpose()
        #for j in range(inner_step_count):
            # Solve system (apply CG or pseudospectral)
        #u_half = eigvec @ (demon @ u_old)  # Project back into normal space
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #u_old = u_new.copy()
        #vv = u_old.copy()
        #ww = vv.copy()
        
        #start_time_diffusion = time.time()

        #for j in range(inner_step_count):
        #    mean_f = np.dot(degree.reshape(1, len(degree)), vv) / np.sum(degree)
        #    ww += 2 * gamma * dt * degree_diag @ (vv - mean_f)
        #    vv = _diffusion_step_eig(ww,eigvec,eigval,dt)

        #start_time_thresholding = time.time()
        # Apply thresholding 
        #u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(vv)
        
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 

        #start_time_stop_criterion = time.time()
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        #stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion))
        
        #n = n+1
    #print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))


    #print('Start Hu, Laurent algorithm')
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)

    start_time_eigendecomposition = time.time()
    #D, V = eigsh(
    #    nor_graph_laplacian,
    #    k=m,
    #    sigma=0,
    #    v0=np.ones((nor_graph_laplacian.shape[0], 1)),
    #    which='SA')
    #print('eigendecomposition uses SA')
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))

    #eigenpair = quimb.linalg.slepc_linalg.eigs_slepc(nor_graph_laplacian, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))
    #print('EPSType is krylovschur')
    #D_sign = eigenpair[0]
    #V_sign = eigenpair[1]

    
    #start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    

    stop_criterion = 10
    n = 0
    u_new = u_init.copy()        
    
    start_time_MBO_iteration = time.time()
    # Perform MBO scheme
    while (n < max_iter) and (stop_criterion > tol):
    #for i in range(50):
        u_old = u_new.copy()
        vv = u_old.copy()

        ww = vv.copy()
        
        #start_time_diffusion = time.time()

        for j in range(inner_step_count):
            mean_f = np.dot(degree.reshape(1, len(degree)), vv) / np.sum(degree)
            ww += 2 * gamma * dt * degree_diag @ (vv - mean_f)
            vv = _diffusion_step_eig(ww,eigvec,eigval,dt)

        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        #u_new = apply_threshold(vv, target_size, thresh_type)
        u_new = _mbo_forward_step_multiclass(vv)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 

        #start_time_stop_criterion = time.time()
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion))
        
        n = n+1
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n


def mbo_modularity_inner_step(num_nodes, num_communities, m, u_init,laplacian_mix, eigval, eigvec, dt, tol,target_size,inner_step_count,
                        max_iter=10000,initial_state_type="random", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    print('Start with MMBO with finite difference')

    #start_time_l_mix = time.time()
    #laplacian_mix = graph_lap + signless_lap
    #print("compute l_{mix}:-- %.3f seconds --" % (time.time() - start_time_l_mix))
    
    start_time_eigendecomposition = time.time()
    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which= "SA",)

    #eigenpair = quimb.linalg.slepc_linalg.eigs_slepc(laplacian_mix, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))
    #print('EPSType is krylovschur')
    #D_sign = eigenpair[0]
    #V_sign = eigenpair[1]
   
    #start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))


    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u_init.copy()
    
    start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
    #for i in range(50):
        u_old = u_new.copy()

        dti = dt / (2 * inner_step_count)

        #start_time_diffusion = time.time()

        demon = sp.sparse.spdiags([1 / (1 + dti * eigval)], [0], m, m) @ eigvec.transpose()

        
        for j in range(inner_step_count):
            u_half = eigvec @ (demon @ u_old)
        
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        u_new = apply_threshold(u_half, target_size, thresh_type)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 
        
        #start_time_stop_criterion = time.time()
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion)) 

        n = n+1
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n




def mbo_modularity_hu_original(num_nodes, num_communities, m, degree, dt, u_init,nor_graph_laplacian, eigval, eigvec, tol, target_size, inner_step_count, 
                            gamma=0.5, max_iter=10000, thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    print('Start Hu, Laurent algorithm')
    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    print('degree shape: ', degree.shape)
    start_time_eigendecomposition = time.time()
    #D, V = eigsh(
    #    nor_graph_laplacian,
    #    k=m,
    #    sigma=0,
    #    v0=np.ones((nor_graph_laplacian.shape[0], 1)),
    #    which='SA')
    #print('eigendecomposition uses SA')
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))

    #eigenpair = quimb.linalg.slepc_linalg.eigs_slepc(nor_graph_laplacian, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))
    #print('EPSType is krylovschur')
    #D_sign = eigenpair[0]
    #V_sign = eigenpair[1]

    
    #start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    

    stop_criterion = 10
    n = 0
    u_new = u_init.copy()        
    
    start_time_MBO_iteration = time.time()
    # Perform MBO scheme
    while (n < max_iter) and (stop_criterion > tol):
    #for i in range(50):
        u_old = u_new.copy()
        vv = u_old.copy()
        #print('vv shape: ', vv.shape)
        ww = vv.copy()
        #aa = degree.reshape(1, len(degree))
        #aa = degree
        #print('aa shape: ', aa.shape)

        #sum_deg = np.sum(degree)
        #sum_deg_int = sum_deg.astype(int)
        #print('sum_deg_int shape: ', type(sum_deg_int))
        #start_time_diffusion = time.time()
        for j in range(inner_step_count):
            #mean_f = (aa @ vv) / sum_deg_int
            mean_f = np.dot(degree.reshape(1, len(degree)), vv) / np.sum(degree)
            ww += 2 * gamma * dt * degree_diag @ (vv - mean_f)
            vv = _diffusion_step_eig(ww,eigvec,eigval,dt)

        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        #u_new = apply_threshold(vv, target_size, thresh_type)
        u_new = _mbo_forward_step_multiclass(vv)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 

        #start_time_stop_criterion = time.time()
        # Stop criterion
        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion))
        
        n = n+1
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n
