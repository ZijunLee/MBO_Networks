import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import time
import sknetwork as skn
from utils import generate_initial_value_multiclass, _diffusion_step_eig, _mbo_forward_step_multiclass, vector_to_labels


"""
    Run the MBO scheme on a graph.
    Parameters
    ----------
    adj_matrix : np.array
        The adjacency matrix of the (unsigned) graph
    m : int
        Number of eigenvalues to use for MBO scheme
    num_communities : int
        Number of communities
    max_iter : int
        Maximum number of iterations
    N_t : int
        Number of iterations for the MBO diffusion loop
    gamma : int
        Resolution parameter
    stopping_condition : bool
        Use the standard stopping condition or the modularity-related stopping condition
    """


def construct_null_model(adj_matrix):
        
    # Compute the degree of adjacency matrix
    degree = np.array(np.sum(adj_matrix, axis=-1)).flatten()
    dergee_di_null = np.expand_dims(degree, axis=-1)

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    total_degree_int = total_degree.astype(int)
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int

    return null_model
    


def adj_to_laplacian_signless_laplacian(adj_matrix):
        
    # Compute the degree of adjacency matrix
    degree = np.array(np.sum(adj_matrix, axis=-1)).flatten()
    degree_diag = np.diag(degree)

    # Compute the number of nodes
    num_nodes = len(degree)

    # Construct Newman--Girvan null model
    #start_time_construct_null_model = time.time()
    dergee_di_null = np.expand_dims(degree, axis=-1)
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    total_degree_int = total_degree.astype(int)
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int
    #time_null_model = time.time() - start_time_construct_null_model
    #print("construct null model:-- %.3f seconds --" % (time_null_model))

    del dergee_di_null


    # compute unnormalized laplacian
    graph_laplacian = degree_diag - adj_matrix    # L_A = D - A
 
    del adj_matrix

    # compute symmetric normalized laplacian 
    degree_inv = np.diag((1 / degree))
    #degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
    sym_graph_laplacian = np.sqrt(degree_inv) @ (graph_laplacian @ np.sqrt(degree_inv))    # obtain L_A_{sym}
    
    # compute random walk normalized Laplacian
    random_walk_lap =  degree_inv @ graph_laplacian


    # compute unnormalized signless laplacian of null model 
    signless_laplacian_null_model = degree_diag + null_model  # Q_P = D + P(null model)
    
    del null_model

    # compute symmetric normalized signless laplacian
    signless_degree_inv = degree_inv
    sym_signless_laplacian = np.sqrt(signless_degree_inv) @ (signless_laplacian_null_model @ np.sqrt(signless_degree_inv))
    rw_signless_lapclacian =  signless_degree_inv @ signless_laplacian_null_model

    return num_nodes, degree, sym_graph_laplacian,random_walk_lap, sym_signless_laplacian, rw_signless_lapclacian
    


def adj_to_modularity_mat(adj_matrix):
    
    # Compute the degree of adjacency matrix
    degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    dergee_di_null = np.expand_dims(degree, axis=-1)


    # Compute the number of nodes
    num_nodes = len(degree)

    ## Construct Newman--Girvan null model
    null_model = np.zeros((len(degree), len(degree)))
    total_degree = np.sum(adj_matrix)
    total_degree_int = total_degree.astype(int)
    null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree_int


    # Compute the modularity matrix: B = W - P  
    modularity_mat = adj_matrix - null_model   

    # Compute the unnormalized laplacian of B^+
    modularity_mat_positive = np.where(modularity_mat > 0, modularity_mat, 0)       # B^+
    degree_modularity_mat_positive = np.array(np.sum(modularity_mat_positive, axis=1)).flatten()
    degree_diag_positive = np.diag(degree_modularity_mat_positive)
    graph_laplacian_positive = degree_diag_positive - modularity_mat_positive      # L_B^+ = D^+ - B^+ 
    
    # Compute symmetric normalized laplacian & random walk normalized Laplacian of B^+
    degree_inv_positive = np.diag((1.0 /degree_modularity_mat_positive))
    sym_graph_laplacian_positive = np.sqrt(degree_inv_positive) @ (graph_laplacian_positive @ np.sqrt(degree_inv_positive))    # obtain L_A_{sym}
    random_walk_nor_lap_positive =  degree_inv_positive @ graph_laplacian_positive


    # Compute the unnormalized laplacian of B^-
    modularity_mat_negative = -np.where(modularity_mat < 0, modularity_mat, 0)      # B^-
    degree_modularity_mat_negative = np.array(np.sum(modularity_mat_negative, axis=1)).flatten()
    degree_diag_negative = np.diag(degree_modularity_mat_negative)
    signless_graph_laplacian_neg = degree_diag_negative + modularity_mat_negative     # Q_B^- = D^- + B^-
        
    # compute symmetric normalized laplacian & random walk normalized Laplacian of B^-
    degree_inv_negative = np.diag((1.0 / degree_modularity_mat_negative))
    sym_signless_laplacian_negative = np.sqrt(degree_inv_negative) @ (signless_graph_laplacian_neg @ np.sqrt(degree_inv_negative))    
    random_walk_signless_lap_negative =  degree_inv_negative @ signless_graph_laplacian_neg

    return num_nodes, degree, sym_graph_laplacian_positive, random_walk_nor_lap_positive, sym_signless_laplacian_negative, random_walk_signless_lap_negative




def MMBO_using_projection(m, degree, eig_val, eig_vec, tol, u_init, adj_mat,
                     gamma=0.5, eps=1, max_iter=10000, stopping_condition='standard'): # inner stepcount is actually important! and can't be set to 1...
    
    #print('Start computing the MMBO scheme using projection on the eigenvectors')

    #laplacian_mix = graph_lap + signless_lap

    # compute eigenvalues and eigenvectors
    #start_time_eigendecomposition = time.time()
    #eig_val, eig_vec = eigsh(laplacian_mix, k=m, which='SA')
    #print("compute eigenvalues and eigenvectors of l_{mix}:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))


    # Initialize parameters
    #start_time_initialize = time.time()
    #u_init = generate_initial_value_multiclass(initial_state_type, n_samples=num_nodes, n_class=num_communities)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    
    
    # Time step selection
    #start_time_timestep_selection = time.time()
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u_init)/eps)/eig_val[0]
    dti = np.sqrt(dtlow*dthigh)
    print('dti--MMBO projection: ', dti)
    #print("compute time step selection:-- %.3f seconds --" % (time.time() - start_time_timestep_selection))
    #dti = dt

    demon = sp.sparse.spdiags([np.exp(- 0.5 * eig_val * dti)],[0],m,m) @ eig_vec.transpose()

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    proxy = 0
    u_new = u_init.copy()
    modularity_score_list = []
    
    
    #start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
        modularity_score_old = proxy
        u_old = u_new.copy()

        # Diffusion step
        #start_time_diffusion = time.time()
        u_half = eig_vec @ (demon @ u_old)
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        

        # Thresholding 
        #start_time_thresholding = time.time()
        u_new = _mbo_forward_step_multiclass(u_half)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding))            
                    
        # Stop criterion
        if stopping_condition == 'standard':
            # using standard stopping condition (w.r.t norm)
            stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        else:
            # using modularity-related stopping condition
            u_new_label = vector_to_labels(u_new)
            modularity_score_new = skn.clustering.modularity(adj_mat,u_new_label,resolution=gamma)      

            proxy = modularity_score_new
            stop_criterion = np.abs(modularity_score_new - modularity_score_old)  
            modularity_score_list.append(modularity_score_new)

        n = n + 1
            
    #print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))
    
    return u_new, n, modularity_score_list




def MMBO_using_finite_differendce(m, degree, eig_val, eig_vec, tol, N_t, u_init, adj_mat,
                        gamma=0.5, eps=1, max_iter=10000, stopping_condition='standard'): # inner stepcount is actually important! and can't be set to 1...
    
    #print('Start the MMBO scheme using finite difference')

    #laplacian_mix = graph_lap + signless_lap
    
    # compute eigenvalues and eigenvectors
    #start_time_eigendecomposition = time.time()
    #eig_val, eig_vec = eigsh(laplacian_mix, k=m, which= "SA")
    #print("compute eigenvalues and eigenvectors of l_{mix}:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))

   
    # Initialize parameters
    #start_time_initialize = time.time()
    #u_init = generate_initial_value_multiclass(initial_state_type, n_samples=num_nodes, n_class=num_communities)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))

    # Time step selection
    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u_init)/eps)/eig_val[0]
    dti = np.sqrt(dtlow*dthigh)
    #dti = dt
    print('dt: ', dti)

    dti = dti / (2 * N_t)
    #print('dti: ', dti)

    demon = sp.sparse.spdiags([1 / (1 + dti * eig_val)], [0], m, m) @ eig_vec.transpose()

    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    proxy = 0
    u_new = u_init.copy()
    modularity_score_list =[]
    
    #start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
        modularity_score_old = proxy
        u_old = u_new.copy()

        # Diffusion step
        #start_time_diffusion = time.time()
        for j in range(N_t):
            u_half = eig_vec @ (demon @ u_old)
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        # Thresholding 
        #start_time_thresholding = time.time()
        u_new = _mbo_forward_step_multiclass(u_half)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 
        
    
        # Stop criterion
        if stopping_condition == 'standard':
            # using standard stopping condition (w.r.t norm)
            stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        else:
            # using modularity-related stopping condition
            u_new_label = vector_to_labels(u_new)
            modularity_score_new = skn.clustering.modularity(adj_mat,u_new_label,resolution=gamma)      

            proxy = modularity_score_new
            stop_criterion = np.abs(modularity_score_new - modularity_score_old)  
            modularity_score_list.append(modularity_score_new)
        
        n = n + 1
            
    #print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n, modularity_score_list




def HU_mmbo_method(num_nodes, degree, eig_val, eig_vec, tol, N_t, u_init, adj_mat,
                    gamma=0.5, eps=1, max_iter=10000, stopping_condition='standard'): 

    degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)

    #start_time_eigendecomposition = time.time()
    #eig_val_hu_sym, eig_vec_hu_sym = eigsh(sym_graph_laplacian + ((2*gamma/np.sum(degree)) * degree @ degree.T), k=m, which='SA')
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition)) 

      
    # Initialize parameters
    #start_time_initialize = time.time()
    #u_init = generate_initial_value_multiclass(initial_state_type, n_samples=num_nodes, n_class=num_communities)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
        

    n = 0
    stop_criterion = 10
    proxy = 0
    u_new = u_init.copy()
    modularity_score_list =[]   

    dtlow = 0.15/((gamma+1)*np.max(degree))
    dthigh = np.log(np.linalg.norm(u_init)/eps)/eig_val[1]
    dti = np.sqrt(dtlow*dthigh) 


    # Perform MBO scheme
    #start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
        modularity_score_old = proxy
        u_old = u_new.copy()
        vv = u_old.copy()
        ww = vv.copy()

        # Diffusion step
        #start_time_diffusion = time.time()
        for j in range(N_t):
            mean_f = np.dot(degree.reshape(1, len(degree)), vv) / np.sum(degree)
            ww += 2 * gamma * dti * degree_diag @ (vv - mean_f)
            vv = _diffusion_step_eig(ww,eig_vec,eig_val,dti)
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        
        # Thresholding 
        #start_time_thresholding = time.time()
        u_new = _mbo_forward_step_multiclass(vv)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 
        
    
        # Stop criterion
        if stopping_condition == 'standard':
            # using standard stopping condition (w.r.t norm)
            stop_criterion = sp.linalg.norm(u_new-u_old) / sp.linalg.norm(u_new)
        else:
            # using modularity-related stopping condition
            u_new_label = vector_to_labels(u_new)
            modularity_score_new = skn.clustering.modularity(adj_mat,u_new_label,resolution=gamma)      

            proxy = modularity_score_new
            stop_criterion = np.abs(modularity_score_new - modularity_score_old)  
            modularity_score_list.append(modularity_score_new)
        
        n = n + 1
            
    #print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n, modularity_score_list
