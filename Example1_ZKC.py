from cgi import print_arguments
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh
import time
import sknetwork as skn
from MMBO_and_HU import MMBO_using_projection, MMBO_using_finite_differendce, adj_to_laplacian_signless_laplacian,HU_mmbo_method, adj_to_modularity_mat
from utils import vector_to_labels, labels_to_vector, purity_score, inverse_purity_score, generate_initial_value_multiclass



# Example 1: Zachary's Karate Club graph (ZKC)

# Parameter setting

num_communities = 2
m = 1 * num_communities
tol = 0
N_t = 3
gamma = 1


# Get Zachary's Karate Club graph (ZKC)
G = nx.karate_club_graph()

# Ground truth of ZKC
gt_membership = [G.nodes[v]['club'] for v in G.nodes()]

gt_number = []
for i in gt_membership:
    if i == "Mr. Hi":
        gt_number.append(1)
    elif i =="Officer":
        gt_number.append(0)    

gt_array = np.asarray(gt_number)        # gt_number is a list
gt_vec = labels_to_vector(gt_array)     # gt_number is a vector


# Convert the graph G into the adjacency matrix
adj_mat_nparray = nx.convert_matrix.to_numpy_array(G)


## Choose adjacency matrices F and H
# Using F = W, H = P
num_nodes, degree, sym_graph_laplacian,random_walk_lap, sym_signless_laplacian, rw_signless_lapclacian = adj_to_laplacian_signless_laplacian(adj_mat_nparray)
L_mix_sym = sym_graph_laplacian + sym_signless_laplacian
L_mix_rw = random_walk_lap + rw_signless_lapclacian

# Using F = B^+, H =B^-
num_nodes, degree, sym_lap_positive_B, rw_lap_positive_B, sym_signless_lap_negative_B, rw_signless_lap_negative_B = adj_to_modularity_mat(adj_mat_nparray)
L_B_sym = sym_lap_positive_B + sym_signless_lap_negative_B
L_B_rw = rw_lap_positive_B + rw_signless_lap_negative_B



# Compute eigenvalues and eigenvectors
start_time_l_mix_sym = time.time()
eig_val_mmbo_sym, eig_vec_mmbo_sym = eigsh(L_mix_sym, k=m, which='SA')
time_eig_l_mix_sym = time.time() - start_time_l_mix_sym


start_time_l_mix_rw = time.time()
eig_val_mmbo_rw, eig_vec_mmbo_rw = eigsh(L_mix_rw, k=m, which='SA')
time_eig_l_mix_rw = time.time() - start_time_l_mix_rw


start_time_l_mix_B_sym = time.time()
eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym = eigsh(L_B_sym, k=m, which='SA')
time_eig_l_mix_B_sym = time.time() - start_time_l_mix_B_sym


start_time_l_mix_B_rw = time.time()
eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw = eigsh(L_B_rw, k=m, which='SA')
time_eig_l_mix_B_rw = time.time() - start_time_l_mix_B_rw


start_time_l_sym = time.time()
eig_val_hu_sym, eig_vec_hu_sym = eigsh(sym_graph_laplacian, k=m, which='SA')
time_eig_l_sym = time.time() - start_time_l_sym
#eig_val_hu_sym = eig_val_hu_sym[1:m+1]
#eig_vec_hu_sym = eig_vec_hu_sym[:, 1:m+1]

start_time_l_rw = time.time()
eig_val_hu_rw, eig_vec_hu_rw = eigsh(random_walk_lap, k=m, which='SA')
time_eig_l_rw = time.time() - start_time_l_rw
#eig_val_hu_rw = eig_val_hu_rw[1:m+1]
#eig_vec_hu_rw = eig_vec_hu_rw[:,1:m+1]




sum_modularity_hu_sym =0
sum_ARI_hu_original_sym = 0
sum_purify_hu_original_sym =0
sum_inverse_purify_hu_original_sym =0
sum_NMI_hu_original_sym =0
sum_num_iteration_HU_sym = 0
sum_time_hu_sym =0

sum_time_hu_rw = 0
sum_num_iter_HU_rw =0
sum_modularity_hu_rw =0
sum_ARI_hu_original_rw =0
sum_purify_hu_original_rw =0
sum_inverse_purify_hu_original_rw =0
sum_NMI_hu_original_rw =0

sum_time_MMBO_projection_sym = 0
sum_num_iteration_MMBO_projection_l_sym = 0 
sum_modularity_MMBO_projection_l_sym = 0
sum_ARI_MMBO_projection_l_sym = 0
sum_purify_MMBO_projection_l_sym = 0
sum_inverse_purify_MMBO_projection_l_sym = 0
sum_NMI_MMBO_projection_l_sym = 0

sum_time_MMBO_projection_rw =0
sum_num_iteration_MMBO_projection_l_rw = 0 
sum_modularity_MMBO_projection_l_rw = 0
sum_ARI_MMBO_projection_l_rw = 0
sum_purify_MMBO_projection_l_rw = 0
sum_inverse_purify_MMBO_projection_l_rw = 0
sum_NMI_MMBO_projection_l_rw = 0

sum_time_MMBO_projection_B_sym =0
sum_num_repeat_mmbo_proj_B_sym =0
sum_modularity_mmbo_proj_B_sym =0
sum_ARI_mmbo_proj_B_sym =0
sum_purify_mmbo_proj_B_sym =0
sum_inverse_purify_mmbo_proj_B_sym =0
sum_NMI_mmbo_proj_B_sym =0

sum_time_MMBO_projection_B_rw =0
sum_num_iteration_mmbo_proj_B_rw =0
sum_modularity_mmbo_proj_B_rw =0
sum_ARI_mmbo_proj_B_rw =0
sum_purify_mmbo_proj_B_rw =0
sum_inverse_purify_mmbo_proj_B_rw =0
sum_NMI_mmbo_proj_B_rw =0


sum_time_MMBO_inner_step_rw =0
sum_num_repeat_inner_nor =0
sum_modularity_1_inner_nor_1=0
sum_ARI_mbo_1_inner_nor_1 =0
sum_purify_mbo_1_inner_nor_1 =0
sum_inverse_purify_mbo_1_inner_nor_1 =0
sum_NMI_mbo_1_inner_nor_1 =0

sum_num_repeat_inner_rw =0
sum_modularity_1_inner_rw =0
sum_ARI_mbo_1_inner_rw =0
sum_purify_mbo_1_inner_rw =0
sum_inverse_purify_mbo_1_inner_rw =0
sum_NMI_mbo_1_inner_rw =0

sum_MMBO_using_finite_difference_B_sym =0
sum_num_repeat_inner_nor_B_sym =0
sum_modularity_mmbo_inner_B_sym =0
sum_ARI_mmbo_inner_B_sym =0
sum_purify_mmbo_inner_B_sym =0
sum_inverse_purify_mmbo_inner_B_sym =0
sum_NMI_mmbo_inner_B_sym =0

sum_time_MMBO_using_finite_difference_sym = 0
sum_num_iteration_MMBO_using_finite_difference_sym = 0
sum_modularity_MMBO_using_finite_difference_sym = 0
sum_ARI_MMBO_using_finite_difference_sym = 0
sum_purify_MMBO_using_finite_difference_sym = 0
sum_inverse_purify_MMBO_using_finite_difference_sym = 0
sum_NMI_MMBO_using_finite_difference_sym = 0


sum_time_MMBO_using_finite_difference_rw = 0
sum_num_iteration_MMBO_using_finite_difference_rw = 0 
sum_modularity_MMBO_using_finite_difference_rw = 0
sum_ARI_MMBO_using_finite_difference_rw = 0
sum_purify_MMBO_using_finite_difference_rw = 0
sum_inverse_purify_MMBO_using_finite_difference_rw = 0
sum_NMI_MMBO_using_finite_difference_rw = 0

sum_time_MMBO_using_finite_difference_B_rw = 0
sum_num_iertation_MMBO_using_finite_difference_B_rw = 0 
sum_modularity_mmbo_inner_B_rw =0
sum_ARI_mmbo_inner_B_rw =0
sum_purify_mmbo_inner_B_rw =0
sum_inverse_purify_mmbo_inner_B_rw =0
sum_NMI_mmbo_inner_B_rw =0


# run the script 20 times 
for _ in range(20):

    start_time_initialize = time.time()
    u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=num_communities)
    time_initialize_u = time.time() - start_time_initialize

    # Hu's method using L_sym
    start_time_HU_sym = time.time()
    u_hu_sym_vector, num_iteration_HU_sym, HU_sym_modularity_list = HU_mmbo_method(num_nodes, degree, eig_val_hu_sym, eig_vec_hu_sym,
                                 tol, N_t, u_init, adj_mat_nparray, gamma=gamma) 
    time_HU_sym = time.time() - start_time_HU_sym
    time_HU_sym = time_eig_l_sym + time_initialize_u + time_HU_sym
    #print('the num_iteration of HU method with L_sym: ', num_iteration_HU_sym)

    u_hu_sym_label = vector_to_labels(u_hu_sym_vector)

    modularity_hu_sym = skn.clustering.modularity(adj_mat_nparray,u_hu_sym_label,resolution=gamma)
    ARI_hu_sym = adjusted_rand_score(u_hu_sym_label, gt_membership)
    purify_hu_sym = purity_score(gt_membership, u_hu_sym_label)
    inverse_purify_hu_sym = inverse_purity_score(gt_membership, u_hu_sym_label)
    NMI_hu_sym = normalized_mutual_info_score(gt_membership, u_hu_sym_label)

    #print('modularity score for HU method: ', modularity_hu_sym)
    #print('ARI for HU method: ', ARI_hu_sym)
    #print('purify for HU method: ', purify_hu_sym)
    #print('inverse purify for HU method: ', inverse_purify_hu_sym)
    #print('NMI for HU method: ', NMI_hu_sym)
    

    sum_time_hu_sym += time_HU_sym
    sum_num_iteration_HU_sym += num_iteration_HU_sym 
    sum_modularity_hu_sym += modularity_hu_sym
    sum_ARI_hu_original_sym += ARI_hu_sym
    sum_purify_hu_original_sym += purify_hu_sym
    sum_inverse_purify_hu_original_sym += inverse_purify_hu_sym
    sum_NMI_hu_original_sym += NMI_hu_sym


    # HU's method --rw
    start_time_HU_rw = time.time()
    u_hu_vector_rw, num_iter_HU_rw, HU_modularity_list_rw = HU_mmbo_method(num_nodes, degree, eig_val_hu_rw, eig_vec_hu_rw,
                                 tol, N_t, u_init, adj_mat_nparray, gamma=gamma) 
    time_HU_rw = time.time() - start_time_HU_rw
    time_HU_rw = time_eig_l_rw + time_initialize_u + time_HU_rw
    #print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
    #print('the num_iteration of HU method: ', num_iter_HU_rw)

    u_hu_label_rw = vector_to_labels(u_hu_vector_rw)

    modu_Hu_rw = skn.clustering.modularity(adj_mat_nparray,u_hu_label_rw,resolution=gamma)
    ARI_Hu_rw = adjusted_rand_score(u_hu_label_rw, gt_membership)
    purify_Hu_rw = purity_score(gt_membership, u_hu_label_rw)
    inverse_purify_Hu_rw = inverse_purity_score(gt_membership, u_hu_label_rw)
    NMI_Hu_rw = normalized_mutual_info_score(gt_membership, u_hu_label_rw)

    #print('HU method --random walk')
    #print('modularity score for HU method: ', modu_Hu_rw)
    #print('ARI for HU method: ', ARI_Hu_rw)
    #print('purify for HU method: ', purify_Hu_rw)
    #print('inverse purify for HU method: ', inverse_purify_Hu_rw)
    #print('NMI for HU method: ', NMI_Hu_rw)
    

    sum_time_hu_rw += time_HU_rw
    sum_num_iter_HU_rw += num_iter_HU_rw 
    sum_modularity_hu_rw += modu_Hu_rw
    sum_ARI_hu_original_rw += ARI_Hu_rw
    sum_purify_hu_original_rw += purify_Hu_rw
    sum_inverse_purify_hu_original_rw += inverse_purify_Hu_rw
    sum_NMI_hu_original_rw += NMI_Hu_rw

    
    # MMBO projection l_sym
    start_time_MMBO_projection_l_sym = time.time()
    u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree,  
                                            eig_val_mmbo_sym, eig_vec_mmbo_sym, tol, u_init, adj_mat_nparray, gamma=gamma) 
    time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
    time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
    #print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

    u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
    modularity_MMBO_projection_l_sym = skn.clustering.modularity(adj_mat_nparray ,u_MMBO_projection_l_sym_label,resolution=gamma)
    ARI_MMBO_projection_l_sym = adjusted_rand_score(u_MMBO_projection_l_sym_label, gt_membership)
    purify_MMBO_projection_l_sym = purity_score(gt_membership, u_MMBO_projection_l_sym_label)
    inverse_purify_MMBO_projection_l_sym = inverse_purity_score(gt_membership, u_MMBO_projection_l_sym_label)
    NMI_MMBO_projection_l_sym = normalized_mutual_info_score(gt_membership, u_MMBO_projection_l_sym_label)

    #print('modularity for MMBO using projection with L_W&P: ', modularity_MMBO_projection_l_sym)
    #print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
    #print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
    #print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
    #print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)


    sum_time_MMBO_projection_sym += time_MMBO_projection_sym
    sum_num_iteration_MMBO_projection_l_sym += num_iteration_MMBO_projection_l_sym 
    sum_modularity_MMBO_projection_l_sym += modularity_MMBO_projection_l_sym
    sum_ARI_MMBO_projection_l_sym += ARI_MMBO_projection_l_sym
    sum_purify_MMBO_projection_l_sym += purify_MMBO_projection_l_sym
    sum_inverse_purify_MMBO_projection_l_sym += inverse_purify_MMBO_projection_l_sym
    sum_NMI_MMBO_projection_l_sym += NMI_MMBO_projection_l_sym


    # MMBO projection l_rw
    start_time_MMBO_projection_l_rw = time.time()
    u_MMBO_projection_l_rw, num_iteration_MMBO_projection_l_rw, MMBO_projection_l_rw_modularity_list = MMBO_using_projection(m, degree,  
                                            eig_val_mmbo_rw, eig_vec_mmbo_rw, tol, u_init, adj_mat_nparray, gamma=gamma) 
    time_MMBO_projection_rw = time.time() - start_time_MMBO_projection_l_rw
    time_MMBO_projection_rw = time_eig_l_mix_rw + time_initialize_u + time_MMBO_projection_rw
    #print('the number of MBO iteration for MMBO using projection with L_W&P_rw: ', num_iteration_MMBO_projection_l_rw)

    u_MMBO_projection_l_rw_label = vector_to_labels(u_MMBO_projection_l_rw)
    modularity_MMBO_projection_l_rw = skn.clustering.modularity(adj_mat_nparray,u_MMBO_projection_l_rw_label,resolution=gamma)
    ARI_MMBO_projection_l_rw = adjusted_rand_score(u_MMBO_projection_l_rw_label, gt_membership)
    purify_MMBO_projection_l_rw = purity_score(gt_membership, u_MMBO_projection_l_rw_label)
    inverse_purify_MMBO_projection_l_rw = inverse_purity_score(gt_membership, u_MMBO_projection_l_rw_label)
    NMI_MMBO_projection_l_rw = normalized_mutual_info_score(gt_membership, u_MMBO_projection_l_rw_label)

    sum_time_MMBO_projection_rw += time_MMBO_projection_sym
    sum_num_iteration_MMBO_projection_l_rw += num_iteration_MMBO_projection_l_rw 
    sum_modularity_MMBO_projection_l_rw += modularity_MMBO_projection_l_rw
    sum_ARI_MMBO_projection_l_rw += ARI_MMBO_projection_l_rw
    sum_purify_MMBO_projection_l_rw += purify_MMBO_projection_l_rw
    sum_inverse_purify_MMBO_projection_l_rw += inverse_purify_MMBO_projection_l_rw
    sum_NMI_MMBO_projection_l_rw += NMI_MMBO_projection_l_rw


    # MMBO projection B_sym
    start_time_MMBO_projection_B_sym = time.time()
    u_mmbo_proj_B_sym, num_iteration_mmbo_proj_B_sym, MMBO_projection_B_sym_modularity_list = MMBO_using_projection(m, degree,  
                                            eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym, tol, u_init, adj_mat_nparray, gamma=gamma) 
    time_MMBO_projection_B_sym = time.time() - start_time_MMBO_projection_B_sym
    time_MMBO_projection_B_sym = time_eig_l_mix_B_sym + time_initialize_u + time_MMBO_projection_B_sym
    #print('the number of MBO iteration for MMBO using projection with L_B_sym: ', num_repeat_mmbo_proj_B_sym)

    u_mmbo_proj_B_sym_label = vector_to_labels(u_mmbo_proj_B_sym)
    modularity_mmbo_proj_B_sym = skn.clustering.modularity(adj_mat_nparray,u_mmbo_proj_B_sym_label,resolution=gamma)
    ARI_mmbo_proj_B_sym = adjusted_rand_score(u_mmbo_proj_B_sym_label, gt_membership)
    purify_mmbo_proj_B_sym = purity_score(gt_membership, u_mmbo_proj_B_sym_label)
    inverse_purify_mmbo_proj_B_sym = inverse_purity_score(gt_membership, u_mmbo_proj_B_sym_label)
    NMI_mmbo_proj_B_sym = normalized_mutual_info_score(gt_membership, u_mmbo_proj_B_sym_label)

    sum_time_MMBO_projection_B_sym += time_MMBO_projection_B_sym
    sum_num_repeat_mmbo_proj_B_sym += num_iteration_mmbo_proj_B_sym 
    sum_modularity_mmbo_proj_B_sym += modularity_mmbo_proj_B_sym
    sum_ARI_mmbo_proj_B_sym += ARI_mmbo_proj_B_sym
    sum_purify_mmbo_proj_B_sym += purify_mmbo_proj_B_sym
    sum_inverse_purify_mmbo_proj_B_sym += inverse_purify_mmbo_proj_B_sym
    sum_NMI_mmbo_proj_B_sym += NMI_mmbo_proj_B_sym

    # MMBO projection B_rw
    start_time_MMBO_projection_B_rw = time.time()
    u_mmbo_proj_B_rw, num_iteration_mmbo_proj_B_rw, MMBO_projection_B_rw_modularity_list = MMBO_using_projection(m, degree,  
                                            eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw, tol, u_init, adj_mat_nparray, gamma=gamma)
    time_MMBO_projection_B_rw = time.time() - start_time_MMBO_projection_B_rw
    time_MMBO_projection_B_sym = time_eig_l_mix_B_rw + time_initialize_u + time_MMBO_projection_B_rw
    #print('the number of MBO iteration for MMBO using projection with L_B_rw: ', num_repeat_mmbo_proj_B_rw)

    u_mmbo_proj_B_rw_label = vector_to_labels(u_mmbo_proj_B_rw)
    modularity_mmbo_proj_B_rw = skn.clustering.modularity(adj_mat_nparray,u_mmbo_proj_B_rw_label,resolution=gamma)
    ARI_mmbo_proj_B_rw = adjusted_rand_score(u_mmbo_proj_B_rw_label, gt_membership)
    purify_mmbo_proj_B_rw = purity_score(gt_membership, u_mmbo_proj_B_rw_label)
    inverse_purify_mmbo_proj_B_rw = inverse_purity_score(gt_membership, u_mmbo_proj_B_rw_label)
    NMI_mmbo_proj_B_rw = normalized_mutual_info_score(gt_membership, u_mmbo_proj_B_rw_label)

    sum_time_MMBO_projection_B_rw += time_MMBO_projection_B_rw
    sum_num_iteration_mmbo_proj_B_rw += num_iteration_mmbo_proj_B_rw 
    sum_modularity_mmbo_proj_B_rw += modularity_mmbo_proj_B_rw
    sum_ARI_mmbo_proj_B_rw += ARI_mmbo_proj_B_rw
    sum_purify_mmbo_proj_B_rw += purify_mmbo_proj_B_rw
    sum_inverse_purify_mmbo_proj_B_rw += inverse_purify_mmbo_proj_B_rw
    sum_NMI_mmbo_proj_B_rw += NMI_mmbo_proj_B_rw


    # MMBO using finite difference L_sym
    start_time_MMBO_using_finite_difference_sym = time.time()
    u_MMBO_using_finite_difference_sym, num_iteration_MMBO_using_finite_difference_sym, MMBO_using_finite_difference_sym_modularity_list = MMBO_using_finite_differendce(m,degree, 
                                        eig_val_mmbo_sym, eig_vec_mmbo_sym, tol, N_t,  u_init, adj_mat_nparray, gamma=gamma) 
    time_MMBO_using_finite_difference_sym = time.time() - start_time_MMBO_using_finite_difference_sym
    time_MMBO_using_finite_difference_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_using_finite_difference_sym
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
    #print('the number of MBO iteration for MMBO using finite difference with L_W&P_sym: ',num_iteration_MMBO_using_finite_difference_sym)

    u_MMBO_using_finite_difference_sym_label = vector_to_labels(u_MMBO_using_finite_difference_sym)
    modularity_MMBO_using_finite_difference_sym = skn.clustering.modularity(adj_mat_nparray,u_MMBO_using_finite_difference_sym_label,resolution=gamma)
    ARI_MMBO_using_finite_difference_sym = adjusted_rand_score(u_MMBO_using_finite_difference_sym_label, gt_membership)
    purify_MMBO_using_finite_difference_sym = purity_score(gt_membership, u_MMBO_using_finite_difference_sym_label)
    inverse_purify_MMBO_using_finite_difference_sym1 = inverse_purity_score(gt_membership, u_MMBO_using_finite_difference_sym_label)
    NMI_MMBO_using_finite_difference_sym = normalized_mutual_info_score(gt_membership, u_MMBO_using_finite_difference_sym_label)
    
    #print('modularity for MMBO using finite difference with L_W&P: ', modularity_MMBO_using_finite_difference_sym)
    #print('ARI for MMBO using finite difference with L_W&P: ', ARI_MMBO_using_finite_difference_sym)
    #print('purify for MMBO using finite difference with L_W&P: ', purify_MMBO_using_finite_difference_sym)
    #print('inverse purify for MMBO using finite difference with L_W&P: ', inverse_purify_MMBO_using_finite_difference_sym1)
    #print('NMI for MMBO using finite difference with L_W&P: ', NMI_MMBO_using_finite_difference_sym)

    sum_time_MMBO_using_finite_difference_sym += time_MMBO_using_finite_difference_sym
    sum_num_iteration_MMBO_using_finite_difference_sym += num_iteration_MMBO_using_finite_difference_sym 
    sum_modularity_MMBO_using_finite_difference_sym += modularity_MMBO_using_finite_difference_sym
    sum_ARI_MMBO_using_finite_difference_sym += ARI_MMBO_using_finite_difference_sym
    sum_purify_MMBO_using_finite_difference_sym += purify_MMBO_using_finite_difference_sym
    sum_inverse_purify_MMBO_using_finite_difference_sym += inverse_purify_MMBO_using_finite_difference_sym1
    sum_NMI_MMBO_using_finite_difference_sym += NMI_MMBO_using_finite_difference_sym


    # MMBO using finite difference L_rw
    start_time_MMBO_using_finite_difference_rw = time.time()
    u_MMBO_using_finite_difference_rw, num_iteration_MMBO_using_finite_difference_rw, MMBO_using_finite_difference_rw_modularity_list = MMBO_using_finite_differendce(m,degree, 
                                        eig_val_mmbo_rw, eig_vec_mmbo_rw, tol, N_t,  u_init, adj_mat_nparray, gamma=gamma)
    time_MMBO_using_finite_difference_rw = time.time() - start_time_MMBO_using_finite_difference_rw
    time_MMBO_using_finite_difference_rw = time_eig_l_mix_rw + time_initialize_u + time_MMBO_using_finite_difference_rw
    #print('the number of MBO iteration for MMBO using inner step with L_W&P_rw: ',num_repeat_inner_rw)

    u_MMBO_using_finite_difference_rw_label = vector_to_labels(u_MMBO_using_finite_difference_rw)
    modularity_MMBO_using_finite_difference_rw = skn.clustering.modularity(adj_mat_nparray,u_MMBO_using_finite_difference_rw_label,resolution=gamma)
    ARI_MMBO_using_finite_difference_rw = adjusted_rand_score(u_MMBO_using_finite_difference_rw_label, gt_membership)
    purify_MMBO_using_finite_difference_rw = purity_score(gt_membership, u_MMBO_using_finite_difference_rw_label)
    inverse_purify_MMBO_using_finite_difference_rw = inverse_purity_score(gt_membership, u_MMBO_using_finite_difference_rw_label)
    NMI_MMBO_using_finite_difference_rw = normalized_mutual_info_score(gt_membership, u_MMBO_using_finite_difference_rw_label)


    #print('modularity for MMBO using inner step with L_W&P_rw: ', modularity_MMBO_using_finite_difference_rw)
    #print('ARI for MMBO using inner step with L_W&P_rw: ', ARI_MMBO_using_finite_difference_rw)
    #print('purify for MMBO using inner step with L_W&P_rw: ', purify_MMBO_using_finite_difference_rw)
    #print('inverse purify for MMBO using inner step with L_W&P_rw: ', inverse_purify_MMBO_using_finite_difference_rw)
    #print('NMI for MMBO using inner step with L_W&P_rw: ', NMI_MMBO_using_finite_difference_rw)

    sum_time_MMBO_using_finite_difference_rw += time_MMBO_using_finite_difference_rw
    sum_num_iteration_MMBO_using_finite_difference_rw += num_iteration_MMBO_using_finite_difference_rw 
    sum_modularity_MMBO_using_finite_difference_rw += modularity_MMBO_using_finite_difference_rw
    sum_ARI_MMBO_using_finite_difference_rw += ARI_MMBO_using_finite_difference_rw
    sum_purify_MMBO_using_finite_difference_rw += purify_MMBO_using_finite_difference_rw
    sum_inverse_purify_MMBO_using_finite_difference_rw += inverse_purify_MMBO_using_finite_difference_rw
    sum_NMI_MMBO_using_finite_difference_rw += NMI_MMBO_using_finite_difference_rw


    # MMBO using finite difference B_sym
    start_time_MMBO_using_finite_difference_B_sym = time.time()
    u_MMBO_using_finite_difference_B_sym, num_iteration_MMBO_using_finite_difference_B_sym, MMBO_using_finite_difference_B_sym_modularity_list = MMBO_using_finite_differendce(m,degree, 
                                        eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym, tol, N_t,  u_init, adj_mat_nparray, gamma=gamma)
    time_start_time_MMBO_using_finite_difference_B_sym = time.time() - start_time_MMBO_using_finite_difference_B_sym
    time_start_time_MMBO_using_finite_difference_B_sym = time_eig_l_mix_B_sym + time_initialize_u + time_start_time_MMBO_using_finite_difference_B_sym
    #print('the number of MBO iteration for MMBO using inner step with L_B_sym: ',num_repeat_inner_nor_B_sym)

    u_MMBO_using_finite_difference_B_sym_label = vector_to_labels(u_MMBO_using_finite_difference_B_sym)
    modularity_MMBO_using_finite_difference_B_sym = skn.clustering.modularity(adj_mat_nparray,u_MMBO_using_finite_difference_B_sym_label,resolution=gamma)
    ARI_MMBO_using_finite_difference_B_sym = adjusted_rand_score(u_MMBO_using_finite_difference_B_sym_label, gt_membership)
    purify_MMBO_using_finite_difference_B_sym = purity_score(gt_membership, u_MMBO_using_finite_difference_B_sym_label)
    inverse_purify_MMBO_using_finite_difference_B_sym = inverse_purity_score(gt_membership, u_MMBO_using_finite_difference_B_sym_label)
    NMI_MMBO_using_finite_difference_B_sym = normalized_mutual_info_score(gt_membership, u_MMBO_using_finite_difference_B_sym_label)

    sum_MMBO_using_finite_difference_B_sym += time_start_time_MMBO_using_finite_difference_B_sym
    sum_num_repeat_inner_nor_B_sym += num_iteration_MMBO_using_finite_difference_B_sym 
    sum_modularity_mmbo_inner_B_sym += modularity_MMBO_using_finite_difference_B_sym
    sum_ARI_mmbo_inner_B_sym += ARI_MMBO_using_finite_difference_B_sym
    sum_purify_mmbo_inner_B_sym += purify_MMBO_using_finite_difference_B_sym
    sum_inverse_purify_mmbo_inner_B_sym += inverse_purify_MMBO_using_finite_difference_B_sym
    sum_NMI_mmbo_inner_B_sym += NMI_MMBO_using_finite_difference_B_sym


    # MMBO using finite difference B_rw
    start_time_MMBO_using_finite_difference_B_rw = time.time()
    u_MMBO_using_finite_difference_B_rw, num_iertation_MMBO_using_finite_difference_B_rw, MMBO_using_finite_difference_B_rw_modularity_list = MMBO_using_finite_differendce(m,degree, 
                                        eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw, tol, N_t,  u_init, adj_mat_nparray, gamma=gamma)
    time_MMBO_using_finite_difference_B_rw = time.time() - start_time_MMBO_using_finite_difference_B_rw
    #print('the number of MBO iteration for MMBO using inner step with L_B_rw: ',num_repeat_inner_B_rw)

    u_MMBO_using_finite_difference_B_rw_label = vector_to_labels(u_MMBO_using_finite_difference_B_rw)
    modularity_MMBO_using_finite_difference_B_rw = skn.clustering.modularity(adj_mat_nparray,u_MMBO_using_finite_difference_B_rw_label,resolution=1)
    ARI_mmbo_inner_B_rwMMBO_using_finite_difference_B_rw = adjusted_rand_score(u_MMBO_using_finite_difference_B_rw_label, gt_membership)
    purify_MMBO_using_finite_difference_B_rw = purity_score(gt_membership, u_MMBO_using_finite_difference_B_rw_label)
    inverse_purifyMMBO_using_finite_difference_B_rw = inverse_purity_score(gt_membership, u_MMBO_using_finite_difference_B_rw_label)
    NMI_MMBO_using_finite_difference_B_rw = normalized_mutual_info_score(gt_membership, u_MMBO_using_finite_difference_B_rw_label)


    sum_time_MMBO_using_finite_difference_B_rw += time_MMBO_using_finite_difference_B_rw
    sum_num_iertation_MMBO_using_finite_difference_B_rw += num_iertation_MMBO_using_finite_difference_B_rw 
    sum_modularity_mmbo_inner_B_rw += modularity_MMBO_using_finite_difference_B_rw
    sum_ARI_mmbo_inner_B_rw += ARI_mmbo_inner_B_rwMMBO_using_finite_difference_B_rw
    sum_purify_mmbo_inner_B_rw += purify_MMBO_using_finite_difference_B_rw
    sum_inverse_purify_mmbo_inner_B_rw += inverse_purifyMMBO_using_finite_difference_B_rw
    sum_NMI_mmbo_inner_B_rw += NMI_MMBO_using_finite_difference_B_rw


print('MMBO using projection L_sym')
average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 20
average_num_iter_MMBO_projection_sym = sum_num_iteration_MMBO_projection_l_sym / 20
average_modularity_MMBO_projection_sym = sum_modularity_MMBO_projection_l_sym / 20
average_ARI_MMBO_projection_sym = sum_ARI_MMBO_projection_l_sym / 20
average_purify_MMBO_projection_sym = sum_purify_MMBO_projection_l_sym / 20
average_inverse_purify_MMBO_projection_sym = sum_inverse_purify_MMBO_projection_l_sym / 20
average_NMI_MMBO_projection_sym = sum_NMI_MMBO_projection_l_sym / 20


print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_sym)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_sym)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_sym)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_sym)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_sym)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_sym)


print('MMBO using projection L_rw')
average_time_MMBO_projection_rw = sum_time_MMBO_projection_rw / 20
average_num_iteration_MMBO_projection_rw = sum_num_iteration_MMBO_projection_l_rw / 20
average_modularity_MMBO_projection_rw = sum_modularity_MMBO_projection_l_rw / 20
average_ARI_MMBO_projection_rw = sum_ARI_MMBO_projection_l_rw / 20
average_purify_MMBO_projection_rw = sum_purify_MMBO_projection_l_rw / 20
average_inverse_purify_MMBO_projection_rw = sum_inverse_purify_MMBO_projection_l_rw / 20
average_NMI_MMBO_projection_rw = sum_NMI_MMBO_projection_l_rw / 20


print('average_time_MMBO_projection_rw: ', average_time_MMBO_projection_rw)
print('average_num_iteration_MMBO_projection_rw: ', average_num_iteration_MMBO_projection_rw)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_rw)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_rw)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_rw)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_rw)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_rw)


print('MMBO using projection B_sym')
average_time_MMBO_projection_B_sym = sum_time_MMBO_projection_B_sym / 20
average_num_iter_MMBO_projection_B_sym = sum_num_repeat_mmbo_proj_B_sym / 20
average_modularity_MMBO_projection_B_sym = sum_modularity_mmbo_proj_B_sym / 20
average_ARI_MMBO_projection_B_sym = sum_ARI_mmbo_proj_B_sym / 20
average_purify_MMBO_projection_B_sym = sum_purify_mmbo_proj_B_sym / 20
average_inverse_purify_MMBO_projection_B_sym = sum_inverse_purify_mmbo_proj_B_sym / 20
average_NMI_MMBO_projection_B_sym = sum_NMI_mmbo_proj_B_sym / 20


print('average_time_MMBO_projection_B_sym: ', average_time_MMBO_projection_B_sym)
print('average_num_iteration_MMBO_projection_B_sym: ', average_num_iter_MMBO_projection_B_sym)
print('average_modularity_MMBO_projection_B_sym: ', average_modularity_MMBO_projection_B_sym)
print('average_ARI_MMBO_projection_B_sym: ', average_ARI_MMBO_projection_B_sym)
print('average_purify_MMBO_projection_B_sym: ', average_purify_MMBO_projection_B_sym)
print('average_inverse_purify_MMBO_projection_B_sym: ', average_inverse_purify_MMBO_projection_B_sym)
print('average_NMI_MMBO_projection_B_sym: ', average_NMI_MMBO_projection_B_sym)


print('MMBO using projection B_rw')
average_time_MMBO_projection_B_rw = sum_time_MMBO_projection_B_rw / 20
average_num_iter_MMBO_projection_B_rw = sum_num_iteration_mmbo_proj_B_rw / 20
average_modularity_MMBO_projection_B_rw = sum_modularity_mmbo_proj_B_rw / 20
average_ARI_MMBO_projection_B_rw = sum_ARI_mmbo_proj_B_rw / 20
average_purify_MMBO_projection_B_rw = sum_purify_mmbo_proj_B_rw / 20
average_inverse_purify_MMBO_projection_B_rw = sum_inverse_purify_mmbo_proj_B_rw / 20
average_NMI_MMBO_projection_B_rw = sum_NMI_mmbo_proj_B_rw / 20


print('average_time_MMBO_projection_B_rw: ', average_time_MMBO_projection_B_rw)
print('average_num_iteration_MMBO_projection_B_rw: ', average_num_iter_MMBO_projection_B_rw)
print('average_modularity_MMBO_projection_B_rw: ', average_modularity_MMBO_projection_B_rw)
print('average_ARI_MMBO_projection_B_rw: ', average_ARI_MMBO_projection_B_rw)
print('average_purify_MMBO_projection_symMMBO_projection_B_rw: ', average_purify_MMBO_projection_B_rw)
print('average_inverse_purify_MMBO_projection_B_rw: ', average_inverse_purify_MMBO_projection_B_rw)
print('average_NMI_MMBO_projection_B_rw: ', average_NMI_MMBO_projection_B_rw)


print('MMBO using finite difference L_sym')
average_time_MMBO_inner_step = sum_time_MMBO_using_finite_difference_sym / 20
average_num_iter_MMBO_inner_step = sum_num_iteration_MMBO_using_finite_difference_sym / 20
average_modularity_MMBO_inner_step = sum_modularity_MMBO_using_finite_difference_sym / 20
average_ARI_MMBO_inner_step = sum_ARI_MMBO_using_finite_difference_sym / 20
average_purify_MMBO_inner_step = sum_purify_MMBO_using_finite_difference_sym / 20
average_inverse_purify_MMBO_inner_step = sum_inverse_purify_MMBO_using_finite_difference_sym / 20
average_NMI_MMBO_inner_step = sum_NMI_MMBO_using_finite_difference_sym / 20


print('average_time_MMBO_using_finite_difference_sym: ', average_time_MMBO_inner_step)
print('average_num_iteration_MMBO_using_finite_difference_sym: ', average_num_iter_MMBO_inner_step)
print('average_modularity_MMBO_using_finite_difference_sym: ', average_modularity_MMBO_inner_step)
print('average_ARI_MMBO_using_finite_difference_sym: ', average_ARI_MMBO_inner_step)
print('average_purify_MMBO_using_finite_difference_sym: ', average_purify_MMBO_inner_step)
print('average_inverse_purify_MMBO_using_finite_difference_sym: ', average_inverse_purify_MMBO_inner_step)
print('average_NMI_MMBO_using_finite_difference_sym: ', average_NMI_MMBO_inner_step)


print('MMBO using finite difference L_rw')
average_time_MMBO_using_finite_difference_rw = sum_time_MMBO_using_finite_difference_rw / 20
average_num_iter_MMBO_inner_step_rw = sum_num_iteration_MMBO_using_finite_difference_rw / 20
average_modularity_MMBO_inner_step_rw = sum_modularity_MMBO_using_finite_difference_rw / 20
average_ARI_MMBO_inner_step_rw = sum_ARI_MMBO_using_finite_difference_rw / 20
average_purify_MMBO_inner_step_rw = sum_purify_MMBO_using_finite_difference_rw / 20
average_inverse_purify_MMBO_inner_step_rw = sum_inverse_purify_MMBO_using_finite_difference_rw / 20
average_NMI_MMBO_inner_step_rw = sum_NMI_MMBO_using_finite_difference_rw / 20

print('average_time_MMBO_using_finite_difference_rw: ', average_time_MMBO_using_finite_difference_rw)
print('average_num_iter_MMBO_using_finite_difference_rw: ', average_num_iter_MMBO_inner_step_rw)
print('average_modularity_MMBO_using_finite_difference_rw: ', average_modularity_MMBO_inner_step_rw)
print('average_ARI_MMBO_using_finite_difference_rw: ', average_ARI_MMBO_inner_step_rw)
print('average_purify_MMBO_using_finite_difference_rw: ', average_purify_MMBO_inner_step_rw)
print('average_inverse_purify_MMBO_using_finite_difference_rw: ', average_inverse_purify_MMBO_inner_step_rw)
print('average_NMI_MMBO_using_finite_difference_rw: ', average_NMI_MMBO_inner_step_rw)


print('MMBO using finite difference B_sym')
average_time_MMBO_using_finite_difference_B_sym = sum_MMBO_using_finite_difference_B_sym / 20
average_num_iter_MMBO_inner_step_B_sym = sum_num_repeat_inner_nor_B_sym / 20
average_modularity_MMBO_inner_step_B_sym = sum_modularity_mmbo_inner_B_sym / 20
average_ARI_MMBO_inner_step_B_sym = sum_ARI_mmbo_inner_B_sym / 20
average_purify_MMBO_inner_step_B_sym = sum_purify_mmbo_inner_B_sym / 20
average_inverse_purify_MMBO_inner_step_B_sym = sum_inverse_purify_mmbo_inner_B_sym / 20
average_NMI_MMBO_inner_step_B_sym = sum_NMI_mmbo_inner_B_sym / 20

print('average_time_MMBO_using_finite_difference_B_sym: ',average_time_MMBO_using_finite_difference_B_sym)
print('average_num_iteration_MMBO_using_finite_difference_B_sym: ', average_num_iter_MMBO_inner_step_B_sym)
print('average_modularity_MMBO_using_finite_difference_B_sym: ', average_modularity_MMBO_inner_step_B_sym)
print('average_ARI_MMBO_using_finite_difference_B_sym: ', average_ARI_MMBO_inner_step_B_sym)
print('average_purify_MMBO_using_finite_difference_B_symp: ', average_purify_MMBO_inner_step_B_sym)
print('average_inverse_purify_MMBO_using_finite_difference_B_sym: ', average_inverse_purify_MMBO_inner_step_B_sym)
print('average_NMI_MMBO_using_finite_difference_B_sym: ', average_NMI_MMBO_inner_step_B_sym)


print('MMBO using finite difference B_rw')
average_time_MMBO_using_finite_difference_B_rw = sum_time_MMBO_using_finite_difference_B_rw /20
average_num_iter_MMBO_inner_step_B_rw = sum_num_iertation_MMBO_using_finite_difference_B_rw / 20
average_modularity_MMBO_inner_step_B_rw = sum_modularity_mmbo_inner_B_rw / 20
average_ARI_MMBO_inner_step_B_rw = sum_ARI_mmbo_inner_B_rw / 20
average_purify_MMBO_inner_step_B_rw = sum_purify_mmbo_inner_B_rw / 20
average_inverse_purify_MMBO_inner_step_B_rw = sum_inverse_purify_mmbo_inner_B_rw / 20
average_NMI_MMBO_inner_step_B_rw = sum_NMI_mmbo_inner_B_rw / 20

print('average_time_MMBO_using_finite_difference_B_rw: ',average_time_MMBO_using_finite_difference_B_rw)
print('average_num_ieration_MMBO_using_finite_difference_B_rw: ', average_num_iter_MMBO_inner_step_B_rw)
print('average_modularity_MMBO_using_finite_difference_B_rw: ', average_modularity_MMBO_inner_step_B_rw)
print('average_ARI_MMBO_using_finite_difference_B_rw: ', average_ARI_MMBO_inner_step_B_rw)
print('average_purify_MMBO_using_finite_difference_B_rw: ', average_purify_MMBO_inner_step_B_rw)
print('average_inverse_purify_MMBO_using_finite_difference_B_rw: ', average_inverse_purify_MMBO_inner_step_B_rw)
print('average_NMI_MMBO_using_finite_difference_B_rw: ', average_NMI_MMBO_inner_step_B_rw)



print('HU method L_sym')
average_time_hu_mbo = sum_time_hu_sym / 20
average_num_iter_HU_sym = sum_num_iteration_HU_sym / 20
average_modularity_hu_sym = sum_modularity_hu_sym / 20
average_ARI_hu_original_sym = sum_ARI_hu_original_sym / 20
average_purify_hu_original_sym = sum_purify_hu_original_sym / 20
average_inverse_purify_hu_original_sym = sum_inverse_purify_hu_original_sym / 20
average_NMI_hu_original_sym = sum_NMI_hu_original_sym / 20


print('average_time_HU_sym: ', average_time_hu_mbo)
print('average_num_iteration_HU_sym: ', average_num_iter_HU_sym)
print('average_modularity_HU_sym: ', average_modularity_hu_sym)
print('average_ARI_HU_sym: ', average_ARI_hu_original_sym)
print('average_purify_HU_sym: ', average_purify_hu_original_sym)
print('average_inverse_purify_HU_sym: ', average_inverse_purify_hu_original_sym)
print('average_NMI_HU_sym: ', average_NMI_hu_original_sym)


print('HU method L_rw')
average_time_hu_mbo = sum_time_hu_rw / 20
average_num_iter_HU_rw = sum_num_iter_HU_rw / 20
average_modularity_hu_rw = sum_modularity_hu_rw / 20
average_ARI_hu_original_rw = sum_ARI_hu_original_rw / 20
average_purify_hu_original_rw = sum_purify_hu_original_rw / 20
average_inverse_purify_hu_original_rw = sum_inverse_purify_hu_original_rw / 20
average_NMI_hu_original_rw = sum_NMI_hu_original_rw / 20


print('average_time_HU_rw: ', average_time_hu_mbo)
print('average_num_iteration_HU_rw: ', average_num_iter_HU_rw)
print('average_modularity_HU_rw: ', average_modularity_hu_rw)
print('average_ARI_HU_rw: ', average_ARI_hu_original_rw)
print('average_purify_HU_rw: ', average_purify_hu_original_rw)
print('average_inverse_purify_HU_rw: ', average_inverse_purify_hu_original_rw)
print('average_NMI_HU_rw: ', average_NMI_hu_original_rw)








# Louvain
sum_time_louvain=0
sum_modularity_louvain =0
sum_ARI_louvain = 0
sum_purity_louvain = 0
sum_inverse_purity_louvain = 0
sum_NMI_louvain = 0

for _ in range(20):
    start_time_louvain = time.time()
    partition_Louvain = nx_comm.louvain_communities(G, resolution=gamma, threshold=tol)
    time_louvain = time.time() - start_time_louvain
    #print("Louvain:-- %.3f seconds --" % (time_louvain))

    partition_louvain_list = [list(x) for x in partition_Louvain]
    partition_louvain_expand = sum(partition_louvain_list, [])
    num_cluster_louvain = []
    for cluster in range(len(partition_louvain_list)):
        for number_louvain in range(len(partition_louvain_list[cluster])):
            num_cluster_louvain.append(cluster)

    louvain_dict = dict(zip(partition_louvain_expand, num_cluster_louvain))
    #louvain_list = list(dict.values(louvain_dict))    #convert a dict to list
    #louvain_array = np.asarray(louvain_list)

    partition_louvain_sort = np.sort(partition_louvain_expand)
    louvain_list_sorted = []
    for louvain_element in partition_louvain_sort:
        louvain_list_sorted.append(louvain_dict[louvain_element])
    louvain_array_sorted = np.asarray(louvain_list_sorted)

    louvain_vec = labels_to_vector(louvain_array_sorted)

    modularity_louvain = skn.clustering.modularity(adj_mat_nparray,louvain_array_sorted,resolution=gamma)
    ARI_louvain = adjusted_rand_score(louvain_array_sorted, gt_membership)
    purify_louvain = purity_score(gt_membership, louvain_array_sorted)
    inverse_purify_louvain = inverse_purity_score(gt_membership, louvain_array_sorted)
    NMI_louvain = normalized_mutual_info_score(gt_membership, louvain_array_sorted)

    sum_time_louvain += time_louvain
    sum_modularity_louvain += modularity_louvain
    sum_ARI_louvain += ARI_louvain
    sum_purity_louvain += purify_louvain
    sum_inverse_purity_louvain += inverse_purify_louvain
    sum_NMI_louvain += NMI_louvain


average_time_louvain = sum_time_louvain / 20
average_modularity_louvain = sum_modularity_louvain / 20
average_ARI_louvain = sum_ARI_louvain / 20
average_purify_louvain = sum_purity_louvain / 20
average_inverse_purify_louvain = sum_inverse_purity_louvain / 20
average_NMI_louvain = sum_NMI_louvain / 20

print('The Louvain method')
print('average_time_louvain: ', average_time_louvain)
print('average_modularity_louvain: ', average_modularity_louvain)
print('average_ARI_louvain: ', average_ARI_louvain)
print('average_purify_louvain: ', average_purify_louvain)
print('average_inverse_purify_louvain: ', average_inverse_purify_louvain)
print('average_NMI_louvain: ', average_NMI_louvain)



# Spectral clustering with k-means
sum_time_sc=0
sum_modularity_sc =0
sum_ARI_spectral_clustering = 0
sum_purify_spectral_clustering = 0
sum_inverse_purify_spectral_clustering = 0
sum_NMI_spectral_clustering = 0

for _ in range(20):
    start_time_spectral_clustering = time.time()
    sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed')
    assignment = sc.fit_predict(adj_mat_nparray)
    time_sc = time.time() - start_time_spectral_clustering
    #print("spectral clustering algorithm:-- %.3f seconds --" % (time_sc))

    ass_vec = labels_to_vector(assignment)

    modularity_spectral_clustering = skn.clustering.modularity(adj_mat_nparray,assignment,resolution=1)
    ARI_spectral_clustering = adjusted_rand_score(assignment, gt_membership)
    purify_spectral_clustering = purity_score(gt_membership, assignment)
    inverse_purify_spectral_clustering = inverse_purity_score(gt_membership, assignment)
    NMI_spectral_clustering = normalized_mutual_info_score(gt_membership, assignment)

    #print('modularity Spectral clustering score: ', modularity_spectral_clustering)
    #print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
    #print('purify for Spectral clustering : ', purify_spectral_clustering)
    #print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
    #print('NMI for Spectral clustering: ', NMI_spectral_clustering)
    
    sum_time_sc += time_sc
    sum_modularity_sc += modularity_spectral_clustering
    sum_ARI_spectral_clustering += ARI_spectral_clustering
    sum_purify_spectral_clustering += purify_spectral_clustering
    sum_inverse_purify_spectral_clustering += inverse_purify_spectral_clustering
    sum_NMI_spectral_clustering += NMI_spectral_clustering

average_time_sc = sum_time_sc / 20
average_modularity_sc = sum_modularity_sc / 20
average_ARI_spectral_clustering = sum_ARI_spectral_clustering / 20
average_purify_spectral_clustering = sum_purify_spectral_clustering / 20
average_inverse_purify_spectral_clustering = sum_inverse_purify_spectral_clustering / 20
average_NMI_spectral_clustering = sum_NMI_spectral_clustering / 20

print('Spectral clustering')
print('average_time_sc: ', average_time_sc)
print('average_modularity_sc: ', average_modularity_sc)
print('average_ARI_spectral_clustering: ', average_ARI_spectral_clustering)
print('average_purify_spectral_clustering: ', average_purify_spectral_clustering)
print('average_inverse_purify_spectral_clustering: ', average_inverse_purify_spectral_clustering)
print('average_NMI_spectral_clustering: ', average_NMI_spectral_clustering)



# CNM algorithm
sum_time_CNM =0
sum_modularity_CNM =0
sum_ARI_CNM = 0
sum_purity_CNM = 0
sum_inverse_purity_CNM = 0
sum_NMI_CNM = 0

for _ in range(20):
    start_time_CNM = time.time()
    partition_CNM = nx_comm.greedy_modularity_communities(G, resolution=gamma)
    time_CNM = time.time() - start_time_CNM
    #print("CNM algorithm:-- %.3f seconds --" % (time.time() - start_time_CNM))

    partition_CNM_list = [list(x) for x in partition_CNM]
    partition_CNM_expand = sum(partition_CNM_list, [])

    num_cluster_CNM = []
    for cluster in range(len(partition_CNM_list)):
        for number_CNM in range(len(partition_CNM_list[cluster])):
            num_cluster_CNM.append(cluster)

    CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))

    partition_CNM_sort = np.sort(partition_CNM_expand)
    CNM_list_sorted = []
    for CNM_element in partition_CNM_sort:
        CNM_list_sorted.append(CNM_dict[CNM_element])
    CNM_array_sorted = np.asarray(CNM_list_sorted)
    
    CNM_vec = labels_to_vector(CNM_array_sorted)

    modularity_CNM = skn.clustering.modularity(adj_mat_nparray,CNM_array_sorted,resolution=gamma)
    ARI_CNM = adjusted_rand_score(CNM_array_sorted, gt_membership)
    purify_CNM = purity_score(gt_membership, CNM_array_sorted)
    inverse_purify_CNM = inverse_purity_score(gt_membership, CNM_array_sorted)
    NMI_CNM = normalized_mutual_info_score(gt_membership, CNM_array_sorted)

    sum_time_CNM += time_CNM
    sum_modularity_CNM += modularity_CNM
    sum_ARI_CNM += ARI_CNM
    sum_purity_CNM += purify_CNM
    sum_inverse_purity_CNM += inverse_purify_CNM
    sum_NMI_CNM += NMI_CNM


average_time_CNM = sum_time_CNM / 20
average_modularity_CNM = sum_modularity_CNM / 20
average_ARI_CNM = sum_ARI_CNM / 20
average_purity_CNM = sum_purity_CNM / 20
average_inverse_purity_CNM = sum_inverse_purity_CNM / 20
average_NMI_CNM = sum_NMI_CNM / 20

print('CNM')
print('average_time_CNM: ', average_time_CNM)
print('average_modularity_CNM: ', average_modularity_CNM)
print('average_ARI_CNM: ', average_ARI_CNM)
print('average_purity_CNM: ', average_purity_CNM)
print('average_inverse_purity_CNM: ', average_inverse_purity_CNM)
print('average_NMI_CNM: ', average_NMI_CNM)


# Girvan-Newman algorithm
sum_time_GN =0
sum_modularity_GN =0
sum_ARI_GN = 0
sum_purify_GN = 0
sum_inverse_purify_GN = 0
sum_NMI_GN = 0

for _ in range(20):
    start_time_GN = time.time()
    partition_GN = nx_comm.girvan_newman(G)
    time_GN = time.time() - start_time_GN
    #print("GN algorithm:-- %.3f seconds --" % (time_GN))

    partition_GN_list = []
    for i in next(partition_GN):
      partition_GN_list.append(list(i))
    partition_GN_expand = sum(partition_GN_list, [])
    num_cluster_GN = []
    for cluster in range(len(partition_GN_list)):
        for number_GN in range(len(partition_GN_list[cluster])):
            num_cluster_GN.append(cluster)

    GN_dict = dict(zip(partition_GN_expand, num_cluster_GN))
    partition_GN_sort = np.sort(partition_GN_expand)
    GN_list_sorted = []
    for GN_element in partition_GN_sort:
        GN_list_sorted.append(GN_dict[GN_element])
    GN_array_sorted = np.asarray(GN_list_sorted)
    
    GN_vec = labels_to_vector(GN_array_sorted)

    modularity_GN = skn.clustering.modularity(adj_mat_nparray,GN_array_sorted,resolution=gamma)
    ARI_GN = adjusted_rand_score(GN_array_sorted, gt_membership)
    purify_GN = purity_score(gt_membership, GN_array_sorted)
    inverse_purify_GN = inverse_purity_score(gt_membership, GN_array_sorted)
    NMI_GN = normalized_mutual_info_score(gt_membership, GN_array_sorted)
    
    sum_time_GN += time_GN
    sum_modularity_GN += modularity_GN
    sum_ARI_GN += ARI_GN
    sum_purify_GN+= purify_GN
    sum_inverse_purify_GN += inverse_purify_GN
    sum_NMI_GN += NMI_GN

average_time_GN = sum_time_GN / 20
average_modularity_GN = sum_modularity_GN / 20
average_ARI_GN = sum_ARI_GN / 20
average_purify_GN = sum_purify_GN / 20
average_inverse_purify_GN= sum_inverse_purify_GN / 20
average_NMI_GN = sum_NMI_GN / 20

print('GN')
print('average_time_GN: ', average_time_GN)
print('average_modularity_GN: ', average_modularity_GN)
print('average_ARI_GN: ', average_ARI_GN)
print('average_purify_GN: ', average_purify_GN)
print('average_inverse_purify_GN: ', average_inverse_purify_GN)
print('average_NMI_GN: ', average_NMI_GN)



# Plot the partitions of ZKC obtained with different algorithms

gt_vec = np.where(gt_vec > 0, gt_vec, 0)
u_MMBO_projection_l_sym = np.where(u_MMBO_projection_l_sym > 0, u_MMBO_projection_l_sym, 0)
u_MMBO_using_finite_difference_sym = np.where(u_MMBO_using_finite_difference_sym > 0, u_MMBO_using_finite_difference_sym, 0)
u_hu_sym_vector = np.where(u_hu_sym_vector > 0, u_hu_sym_vector, 0)
ass_vec = np.where(ass_vec > 0, ass_vec, 0)
louvain_vec = np.where(louvain_vec > 0, louvain_vec, 0)
CNM_vec = np.where(CNM_vec > 0, CNM_vec, 0)


#colors1 = ["#FF0000", "#0000FF"]
colors2 = ["#458B74", "#FF9912", "#FF0000", "#0000FF"]
colors3 = ["#FF0000", "#FF9912"]

fig, axes = plt.subplots(8,1,figsize=(13, 45))
axs = axes.flatten()
loc = nx.spring_layout(G)

for image in range(8):
    # Original ZKC visualization without partition
    nx.draw(G, with_labels = True, node_size=1000, edge_color="black", pos=loc, ax = axes[image])
    
    # Ground truth
    for k in range(2):
        nodes = np.argwhere(gt_vec[:, k])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[k], pos=loc, ax = axes[1])

    # Partitioning ZKC using the MMBO scheme with projection
    for p in range(2):
        nodes = np.argwhere(u_MMBO_projection_l_sym[:, p])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[p], pos=loc, ax = axes[2])
    
    # Partitioning ZKC using the MMBO scheme with finite difference
    for i in range(2):
        nodes = np.argwhere(u_MMBO_using_finite_difference_sym[:, i])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[i], pos=loc, ax = axes[3])
    
    # Partitioning ZKC using the Hu's method
    for q in range(2):
        nodes = np.argwhere(u_hu_sym_vector[:, q])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[q], pos=loc, ax = axes[4])

    # Partitioning ZKC using Spectral Clustering
    for k in range(2):
        nodes = np.argwhere(ass_vec[:, k])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[k], pos=loc, ax = axes[5])

    # Partitioning ZKC using the Louvain method
    for l in range(4):
        nodes = np.argwhere(louvain_vec[:,l])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors2[l], pos=loc,ax = axes[6])

    # Partitioning ZKC using the CNM algorithm
    for r in range(3):
        nodes = np.argwhere(CNM_vec[:,r])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors2[r], pos=loc,ax = axes[7])

plt.savefig('ZKC.png')
plt.show()







