from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import random
from sklearn.decomposition import PCA
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import time
from community import community_louvain
from Nystrom_extension_QR import nystrom_QR_l_sym, nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
import graphlearning as gl
import cdlib
from cdlib.algorithms import louvain
from cdlib import evaluation, NodeClustering
from MMBO_and_HU import MMBO_using_projection, MMBO_using_finite_differendce,HU_mmbo_method
from utils import vector_to_labels, labels_to_vector, label_to_dict, dict_to_list_set, purity_score, inverse_purity_score, generate_initial_value_multiclass
from Nystrom_QR_test import nystrom_QR_l_mix_sym_rw_ER_null, nystrom_QR_l_mix_B_sym_rw_ER_null



# Example 2: MNIST (with Erdős–Rényi model)

# Parameter setting

# num_communities is found by Louvain
# choose m = num_communities
tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 0.5
tau = 0.02
num_nodes = 70000

# Load MNIST data, ground truth, and build 10-nearest neighbor weight matrix
data, gt_labels = gl.datasets.load('mnist')
gt_vec = labels_to_vector(gt_labels)

gt_labels_list = list(gt_labels)

# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_labels_list)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_labels_list))


pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)
W = gl.weightmatrix.knn(Z_training, 10, symmetrize=True)
G = nx.convert_matrix.from_scipy_sparse_matrix(W)



# First run the Louvain method in order to get the number of clusters
sum_louvain_cluster =0
sum_time_louvain=0
sum_modularity_louvain =0
sum_ER_modularity_louvain =0
sum_ARI_louvain = 0
sum_purity_louvain = 0
sum_inverse_purity_louvain = 0
sum_NMI_louvain = 0



for _ in range(20):
    start_time_louvain = time.time()
    partition_Louvain = community_louvain.best_partition(G, resolution=gamma)    # returns a dict
    time_louvain = time.time() - start_time_louvain
    #print("Louvain:-- %.3f seconds --" % (time_louvain))

    louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
    louvain_array = np.asarray(louvain_list)
    louvain_cluster = len(np.unique(louvain_array))

    louvain_partition_list = dict_to_list_set(partition_Louvain)
    louvain_communities = NodeClustering(louvain_partition_list, graph=None)


    ER_modularity_louvain = evaluation.erdos_renyi_modularity(G,louvain_communities)[2]
    #modularity_louvain = evaluation.newman_girvan_modularity(G,louvain_communities)[2]
    #modularity_louvain = skn.clustering.modularity(W,louvain_array,resolution=gamma)
    ARI_louvain = adjusted_rand_score(louvain_array, gt_labels)
    purify_louvain = purity_score(gt_labels, louvain_array)
    inverse_purify_louvain = inverse_purity_score(gt_labels, louvain_array)
    NMI_louvain = normalized_mutual_info_score(gt_labels, louvain_array)

    sum_louvain_cluster += louvain_cluster
    sum_time_louvain += time_louvain
    #sum_modularity_louvain += modularity_louvain
    sum_ER_modularity_louvain += ER_modularity_louvain
    sum_ARI_louvain += ARI_louvain
    sum_purity_louvain += purify_louvain
    sum_inverse_purity_louvain += inverse_purify_louvain
    sum_NMI_louvain += NMI_louvain

average_louvain_cluster = sum_louvain_cluster / 20
average_time_louvain = sum_time_louvain / 20
#average_modularity_louvain = sum_modularity_louvain / 20
average_ER_modularity_louvain = sum_ER_modularity_louvain / 20
average_ARI_louvain = sum_ARI_louvain / 20
average_purify_louvain = sum_purity_louvain / 20
average_inverse_purify_louvain = sum_inverse_purity_louvain / 20
average_NMI_louvain = sum_NMI_louvain / 20


print('average_time_louvain: ', average_time_louvain)
#print('average_modularity_louvain: ', average_modularity_louvain)
print('average_ER_modularity_louvain: ', average_ER_modularity_louvain)
print('average_ARI_louvain: ', average_ARI_louvain)
print('average_purify_louvain: ', average_purify_louvain)
print('average_inverse_purify_louvain: ', average_inverse_purify_louvain)
print('average_NMI_louvain: ', average_NMI_louvain)


num_communities  = round(average_louvain_cluster)
m = num_communities


# Compute the eigenvalues and eigenvectors of L_sym and L_rw
eigenvalues_sym, eigenvectors_sym, rw_left_eigvec, order_raw_data_HU, index_HU, time_eig_l_sym, time_eig_l_rw = nystrom_QR_l_sym(Z_training, num_nystrom=500, tau=tau)
eig_val_HU_sym = np.squeeze(eigenvalues_sym[:m])
eig_vec_HU_sym = eigenvectors_sym[:,:m]
eig_vec_HU_rw = rw_left_eigvec[:,:m]

gt_labels_HU = gt_labels[index_HU]
W_HU = gl.weightmatrix.knn(order_raw_data_HU, 10)
degree_W_HU = np.array(np.sum(W_HU, axis=-1)).flatten()


# Compute the eigenvalues and eigenvectors of L_mix_sym and L_mix_rw
eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw_ER_null(Z_training, num_nystrom=500, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]


gt_labels_MMBO = gt_labels[index_MMBO]
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10)
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()



eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym, eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw, order_raw_data_B, index_B, time_eig_B_sym, time_eig_B_rw = nystrom_QR_l_mix_B_sym_rw_ER_null(Z_training, num_nystrom=500, tau=tau)
D_mmbo_B_sym = np.squeeze(eig_val_mmbo_B_sym[:m])
V_mmbo_B_sym = eig_vec_mmbo_B_sym[:,:m]
D_mmbo_B_rw = np.squeeze(eig_val_mmbo_B_rw[:m])
V_mmbo_B_rw = eig_vec_mmbo_B_rw[:,:m]


gt_labels_B = gt_labels[index_B]
W_B = gl.weightmatrix.knn(order_raw_data_B, 10)
degree_W_B = np.array(np.sum(W_B, axis=-1)).flatten()


sum_ER_modularity_hu_sym =0
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
sum_ER_modularity_hu_rw =0
sum_ARI_hu_original_rw =0
sum_purify_hu_original_rw =0
sum_inverse_purify_hu_original_rw =0
sum_NMI_hu_original_rw =0


sum_time_MMBO_projection_sym = 0
sum_num_iteration_MMBO_projection_l_sym = 0 
sum_modularity_MMBO_projection_l_sym = 0
sum_ER_modularity_MMBO_projection_l_sym =0
sum_ARI_MMBO_projection_l_sym = 0
sum_purify_MMBO_projection_l_sym = 0
sum_inverse_purify_MMBO_projection_l_sym = 0
sum_NMI_MMBO_projection_l_sym = 0


sum_time_MMBO_projection_rw =0
sum_num_iteration_MMBO_projection_l_rw = 0 
sum_modularity_MMBO_projection_l_rw = 0
sum_ER_modularity_MMBO_projection_l_rw =0
sum_ARI_MMBO_projection_l_rw = 0
sum_purify_MMBO_projection_l_rw = 0
sum_inverse_purify_MMBO_projection_l_rw = 0
sum_NMI_MMBO_projection_l_rw = 0


sum_time_MMBO_projection_B_sym =0
sum_num_repeat_mmbo_proj_B_sym =0
sum_modularity_mmbo_proj_B_sym =0
sum_ER_modularity_mmbo_proj_B_sym =0
sum_ARI_mmbo_proj_B_sym =0
sum_purify_mmbo_proj_B_sym =0
sum_inverse_purify_mmbo_proj_B_sym =0
sum_NMI_mmbo_proj_B_sym =0

sum_time_MMBO_projection_B_rw =0
sum_num_iteration_mmbo_proj_B_rw =0
sum_modularity_mmbo_proj_B_rw =0
sum_ER_modularity_mmbo_proj_B_rw =0
sum_ARI_mmbo_proj_B_rw =0
sum_purify_mmbo_proj_B_rw =0
sum_inverse_purify_mmbo_proj_B_rw =0
sum_NMI_mmbo_proj_B_rw =0



sum_MMBO_using_finite_difference_B_sym =0
sum_num_repeat_inner_nor_B_sym =0
sum_modularity_mmbo_inner_B_sym =0
sum_ER_modularity_mmbo_inner_B_sym =0
sum_ARI_mmbo_inner_B_sym =0
sum_purify_mmbo_inner_B_sym =0
sum_inverse_purify_mmbo_inner_B_sym =0
sum_NMI_mmbo_inner_B_sym =0

sum_time_MMBO_using_finite_difference_sym = 0
sum_num_iteration_MMBO_using_finite_difference_sym = 0
sum_modularity_MMBO_using_finite_difference_sym = 0
sum_ER_modularity_MMBO_using_finite_difference_sym =0
sum_ARI_MMBO_using_finite_difference_sym = 0
sum_purify_MMBO_using_finite_difference_sym = 0
sum_inverse_purify_MMBO_using_finite_difference_sym = 0
sum_NMI_MMBO_using_finite_difference_sym = 0


sum_time_MMBO_using_finite_difference_rw = 0
sum_num_iteration_MMBO_using_finite_difference_rw = 0 
sum_modularity_MMBO_using_finite_difference_rw = 0
sum_ER_modularity_MMBO_using_finite_difference_rw =0
sum_ARI_MMBO_using_finite_difference_rw = 0
sum_purify_MMBO_using_finite_difference_rw = 0
sum_inverse_purify_MMBO_using_finite_difference_rw = 0
sum_NMI_MMBO_using_finite_difference_rw = 0

sum_time_MMBO_using_finite_difference_B_rw = 0
sum_num_iertation_MMBO_using_finite_difference_B_rw = 0 
sum_modularity_mmbo_inner_B_rw =0
sum_ER_modularity_mmbo_inner_B_rw =0
sum_ARI_mmbo_inner_B_rw =0
sum_purify_mmbo_inner_B_rw =0
sum_inverse_purify_mmbo_inner_B_rw =0
sum_NMI_mmbo_inner_B_rw =0



# run the script 20 times using the modularity − related stopping condition
for _ in range(20):

    start_time_initialize = time.time()
    # Unsupervised
    #print('Unsupervised')
    #u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)

    # 10% supervised
    print('10% supervised')
    expand_zero_columns = np.zeros((num_nodes, num_communities - 10))
    gt_vec = np.append(gt_vec, expand_zero_columns, axis=1)

    u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)

    row_numbers = range(0, len(gt_labels))
    Rs = random.sample(row_numbers, 7000)

    for i in Rs:
        u_init[i,:] = gt_vec[i,:]

    time_initialize_u = time.time() - start_time_initialize


    start_time_HU_sym = time.time()
    u_hu_sym_vector, num_iteration_HU_sym, HU_sym_modularity_list = HU_mmbo_method(num_nodes, degree_W_HU, eig_val_HU_sym, eig_vec_HU_sym,
                                 modularity_tol, N_t, u_init, W_HU, gamma=gamma, stopping_condition='modularity') 
    time_HU_sym = time.time() - start_time_HU_sym
    time_HU_sym = time_eig_l_sym + time_initialize_u + time_HU_sym
    #print('the num_iteration of HU method with L_sym: ', num_iteration_HU_sym)

    u_hu_sym_label = vector_to_labels(u_hu_sym_vector)
    u_hu_sym_dict = label_to_dict(u_hu_sym_label)
    u_hu_sym_list = dict_to_list_set(u_hu_sym_dict)
    u_hu_sym_communities = NodeClustering(u_hu_sym_list, graph=None)

    ER_modularity_hu_sym = evaluation.erdos_renyi_modularity(G, u_hu_sym_communities)[2]
    #modularity_hu_sym = evaluation.newman_girvan_modularity(G,u_hu_sym_communities)[2]
    #modularity_hu_sym = skn.clustering.modularity(W_HU, u_hu_sym_label,resolution=gamma)
    ARI_hu_sym = adjusted_rand_score(u_hu_sym_label, gt_labels_HU)
    purify_hu_sym = purity_score(gt_labels_HU, u_hu_sym_label)
    inverse_purify_hu_sym = inverse_purity_score(gt_labels_HU, u_hu_sym_label)
    NMI_hu_sym = normalized_mutual_info_score(gt_labels_HU, u_hu_sym_label)

    #print('modularity score for HU method: ', modularity_hu_sym)
    #print('ARI for HU method: ', ARI_hu_sym)
    #print('purify for HU method: ', purify_hu_sym)
    #print('inverse purify for HU method: ', inverse_purify_hu_sym)
    #print('NMI for HU method: ', NMI_hu_sym)
    

    sum_time_hu_sym += time_HU_sym
    sum_num_iteration_HU_sym += num_iteration_HU_sym 
    #sum_modularity_hu_sym += modularity_hu_sym
    sum_ER_modularity_hu_sym += ER_modularity_hu_sym
    sum_ARI_hu_original_sym += ARI_hu_sym
    sum_purify_hu_original_sym += purify_hu_sym
    sum_inverse_purify_hu_original_sym += inverse_purify_hu_sym
    sum_NMI_hu_original_sym += NMI_hu_sym


    # HU's method --rw
    start_time_HU_rw = time.time()
    u_hu_vector_rw, num_iter_HU_rw, HU_modularity_list_rw = HU_mmbo_method(num_nodes, degree_W_HU, eig_val_HU_sym, eig_vec_HU_rw,
                                 modularity_tol, N_t, u_init, W_HU, gamma=gamma, stopping_condition='modularity') 
    time_HU_rw = time.time() - start_time_HU_rw
    time_HU_rw = time_eig_l_rw + time_initialize_u + time_HU_rw
    #print('the num_iteration of HU method: ', num_iter_HU_rw)

    u_hu_label_rw = vector_to_labels(u_hu_vector_rw)
    u_hu_rw_dict = label_to_dict(u_hu_label_rw)
    u_hu_rw_list = dict_to_list_set(u_hu_rw_dict)
    u_hu_rw_communities = NodeClustering(u_hu_rw_list, graph=None)

    ER_modu_Hu_rw = evaluation.erdos_renyi_modularity(G, u_hu_rw_communities)[2]
    #modu_Hu_rw = evaluation.newman_girvan_modularity(G, u_hu_rw_communities)[2]
    #modu_Hu_rw = skn.clustering.modularity(gt_labels_HU,u_hu_label_rw,resolution=gamma)
    ARI_Hu_rw = adjusted_rand_score(u_hu_label_rw, gt_labels_HU)
    purify_Hu_rw = purity_score(gt_labels_HU, u_hu_label_rw)
    inverse_purify_Hu_rw = inverse_purity_score(gt_labels_HU, u_hu_label_rw)
    NMI_Hu_rw = normalized_mutual_info_score(gt_labels_HU, u_hu_label_rw)

    #print('HU method --random walk')
    #print('modularity score for HU method: ', modu_Hu_rw)
    #print('ARI for HU method: ', ARI_Hu_rw)
    #print('purify for HU method: ', purify_Hu_rw)
    #print('inverse purify for HU method: ', inverse_purify_Hu_rw)
    #print('NMI for HU method: ', NMI_Hu_rw)
    

    sum_time_hu_rw += time_HU_rw
    sum_num_iter_HU_rw += num_iter_HU_rw 
    #sum_modularity_hu_rw += modu_Hu_rw
    sum_ER_modularity_hu_rw += ER_modu_Hu_rw
    sum_ARI_hu_original_rw += ARI_Hu_rw
    sum_purify_hu_original_rw += purify_Hu_rw
    sum_inverse_purify_hu_original_rw += inverse_purify_Hu_rw
    sum_NMI_hu_original_rw += NMI_Hu_rw

 
    # MMBO projection L_mix_sym
    start_time_MMBO_projection_l_sym = time.time()
    u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
    time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
    time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
    #print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

    u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
    u_MMBO_projection_l_sym_dict = label_to_dict(u_MMBO_projection_l_sym_label)
    u_MMBO_projection_l_sym_list = dict_to_list_set(u_MMBO_projection_l_sym_dict)
    u_MMBO_projection_l_sym_coms = NodeClustering(u_MMBO_projection_l_sym_list, graph=None)
    
    ER_modularity_MMBO_projection_l_sym = evaluation.erdos_renyi_modularity(G,u_MMBO_projection_l_sym_coms)[2]
    #modularity_MMBO_projection_l_sym = evaluation.newman_girvan_modularity(G,u_MMBO_projection_l_sym_coms)[2]
    #modularity_MMBO_projection_l_sym = skn.clustering.modularity(W_MMBO ,u_MMBO_projection_l_sym_label,resolution=gamma)
    ARI_MMBO_projection_l_sym = adjusted_rand_score(u_MMBO_projection_l_sym_label, gt_labels_MMBO)
    purify_MMBO_projection_l_sym = purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
    inverse_purify_MMBO_projection_l_sym = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
    NMI_MMBO_projection_l_sym = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)

    #print('modularity for MMBO using projection with L_W&P: ', modularity_MMBO_projection_l_sym)
    #print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
    #print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
    #print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
    #print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)


    sum_time_MMBO_projection_sym += time_MMBO_projection_sym
    sum_num_iteration_MMBO_projection_l_sym += num_iteration_MMBO_projection_l_sym 
    #sum_modularity_MMBO_projection_l_sym += modularity_MMBO_projection_l_sym
    sum_ER_modularity_MMBO_projection_l_sym += ER_modularity_MMBO_projection_l_sym
    sum_ARI_MMBO_projection_l_sym += ARI_MMBO_projection_l_sym
    sum_purify_MMBO_projection_l_sym += purify_MMBO_projection_l_sym
    sum_inverse_purify_MMBO_projection_l_sym += inverse_purify_MMBO_projection_l_sym
    sum_NMI_MMBO_projection_l_sym += NMI_MMBO_projection_l_sym


    # MMBO projection L_mix_rw
    start_time_MMBO_projection_l_rw = time.time()
    u_MMBO_projection_l_rw, num_iteration_MMBO_projection_l_rw, MMBO_projection_l_rw_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_rw, V_mmbo_rw, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
    time_MMBO_projection_rw = time.time() - start_time_MMBO_projection_l_rw
    time_MMBO_projection_rw = time_eig_l_mix_rw + time_initialize_u + time_MMBO_projection_rw
    #print('the number of MBO iteration for MMBO using projection with L_W&P_rw: ', num_iteration_MMBO_projection_l_rw)

    u_MMBO_projection_l_rw_label = vector_to_labels(u_MMBO_projection_l_rw)
    u_MMBO_projection_l_rw_dict = label_to_dict(u_MMBO_projection_l_rw_label)
    u_MMBO_projection_l_rw_list = dict_to_list_set(u_MMBO_projection_l_rw_dict)
    u_MMBO_projection_l_rw_coms = NodeClustering(u_MMBO_projection_l_rw_list, graph=None)
    
    ER_modularity_MMBO_projection_l_rw = evaluation.erdos_renyi_modularity(G, u_MMBO_projection_l_rw_coms)[2]
    #modularity_MMBO_projection_l_rw = evaluation.newman_girvan_modularity(G, u_MMBO_projection_l_rw_coms)[2]
    #modularity_MMBO_projection_l_rw = skn.clustering.modularity(W_MMBO,u_MMBO_projection_l_rw_label,resolution=gamma)
    ARI_MMBO_projection_l_rw = adjusted_rand_score(u_MMBO_projection_l_rw_label, gt_labels_MMBO)
    purify_MMBO_projection_l_rw = purity_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)
    inverse_purify_MMBO_projection_l_rw = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)
    NMI_MMBO_projection_l_rw = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)

    sum_time_MMBO_projection_rw += time_MMBO_projection_sym
    sum_num_iteration_MMBO_projection_l_rw += num_iteration_MMBO_projection_l_rw 
    #sum_modularity_MMBO_projection_l_rw += modularity_MMBO_projection_l_rw
    sum_ER_modularity_MMBO_projection_l_rw += ER_modularity_MMBO_projection_l_rw
    sum_ARI_MMBO_projection_l_rw += ARI_MMBO_projection_l_rw
    sum_purify_MMBO_projection_l_rw += purify_MMBO_projection_l_rw
    sum_inverse_purify_MMBO_projection_l_rw += inverse_purify_MMBO_projection_l_rw
    sum_NMI_MMBO_projection_l_rw += NMI_MMBO_projection_l_rw



    # MMBO projection B_sym
    start_time_MMBO_projection_B_sym = time.time()
    u_mmbo_proj_B_sym, num_iteration_mmbo_proj_B_sym, MMBO_projection_B_sym_modularity_list = MMBO_using_projection(m, degree_W_B,  
                                            D_mmbo_B_sym, V_mmbo_B_sym, modularity_tol, u_init, W_B, gamma=gamma, stopping_condition='modularity') 
    time_MMBO_projection_B_sym = time.time() - start_time_MMBO_projection_B_sym
    time_MMBO_projection_B_sym = time_eig_B_sym + time_initialize_u + time_MMBO_projection_B_sym
    #print('the number of MBO iteration for MMBO using projection with L_B_sym: ', num_repeat_mmbo_proj_B_sym)

    u_mmbo_proj_B_sym_label = vector_to_labels(u_mmbo_proj_B_sym)
    u_mmbo_proj_B_sym_dict = label_to_dict(u_mmbo_proj_B_sym_label)
    u_mmbo_proj_B_sym_list = dict_to_list_set(u_mmbo_proj_B_sym_dict)
    u_mmbo_proj_B_sym_coms = NodeClustering(u_mmbo_proj_B_sym_list, graph=None)
    
    ER_modularity_mmbo_proj_B_sym = evaluation.erdos_renyi_modularity(G, u_mmbo_proj_B_sym_coms)[2]
    #modularity_mmbo_proj_B_sym = evaluation.newman_girvan_modularity(G, u_mmbo_proj_B_sym_coms)[2]
    #modularity_mmbo_proj_B_sym = skn.clustering.modularity(W_B,u_mmbo_proj_B_sym_label,resolution=gamma)
    ARI_mmbo_proj_B_sym = adjusted_rand_score(u_mmbo_proj_B_sym_label, gt_labels_B)
    purify_mmbo_proj_B_sym = purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
    inverse_purify_mmbo_proj_B_sym = inverse_purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
    NMI_mmbo_proj_B_sym = normalized_mutual_info_score(gt_labels_B, u_mmbo_proj_B_sym_label)

    sum_time_MMBO_projection_B_sym += time_MMBO_projection_B_sym
    sum_num_repeat_mmbo_proj_B_sym += num_iteration_mmbo_proj_B_sym 
    #sum_modularity_mmbo_proj_B_sym += modularity_mmbo_proj_B_sym
    sum_ER_modularity_mmbo_proj_B_sym += ER_modularity_mmbo_proj_B_sym
    sum_ARI_mmbo_proj_B_sym += ARI_mmbo_proj_B_sym
    sum_purify_mmbo_proj_B_sym += purify_mmbo_proj_B_sym
    sum_inverse_purify_mmbo_proj_B_sym += inverse_purify_mmbo_proj_B_sym
    sum_NMI_mmbo_proj_B_sym += NMI_mmbo_proj_B_sym


    # MMBO projection B_rw
    start_time_MMBO_projection_B_rw = time.time()
    u_mmbo_proj_B_rw, num_iteration_mmbo_proj_B_rw, MMBO_projection_B_rw_modularity_list = MMBO_using_projection(m, degree_W_B,  
                                            D_mmbo_B_rw, V_mmbo_B_rw, modularity_tol, u_init, W_B, gamma=gamma, stopping_condition='modularity')
    time_MMBO_projection_B_rw = time.time() - start_time_MMBO_projection_B_rw
    time_MMBO_projection_B_sym = time_eig_B_rw + time_initialize_u + time_MMBO_projection_B_rw
    #print('the number of MBO iteration for MMBO using projection with L_B_rw: ', num_repeat_mmbo_proj_B_rw)

    u_mmbo_proj_B_rw_label = vector_to_labels(u_mmbo_proj_B_rw)
    u_mmbo_proj_B_rw_dict = label_to_dict(u_mmbo_proj_B_rw_label)
    u_mmbo_proj_B_rw_list = dict_to_list_set(u_mmbo_proj_B_rw_dict)
    u_mmbo_proj_B_rw_coms = NodeClustering(u_mmbo_proj_B_rw_list, graph=None)
    
    ER_modularity_mmbo_proj_B_rw = evaluation.erdos_renyi_modularity(G, u_mmbo_proj_B_rw_coms)[2]
    #modularity_mmbo_proj_B_rw = evaluation.newman_girvan_modularity(G, u_mmbo_proj_B_rw_coms)[2]
    #modularity_mmbo_proj_B_rw = skn.clustering.modularity(W_B,u_mmbo_proj_B_rw_label,resolution=gamma)
    ARI_mmbo_proj_B_rw = adjusted_rand_score(u_mmbo_proj_B_rw_label, gt_labels_B)
    purify_mmbo_proj_B_rw = purity_score(gt_labels_B, u_mmbo_proj_B_rw_label)
    inverse_purify_mmbo_proj_B_rw = inverse_purity_score(gt_labels_B, u_mmbo_proj_B_rw_label)
    NMI_mmbo_proj_B_rw = normalized_mutual_info_score(gt_labels_B, u_mmbo_proj_B_rw_label)

    sum_time_MMBO_projection_B_rw += time_MMBO_projection_B_rw
    sum_num_iteration_mmbo_proj_B_rw += num_iteration_mmbo_proj_B_rw 
    #sum_modularity_mmbo_proj_B_rw += modularity_mmbo_proj_B_rw
    sum_ER_modularity_mmbo_proj_B_rw += ER_modularity_mmbo_proj_B_rw
    sum_ARI_mmbo_proj_B_rw += ARI_mmbo_proj_B_rw
    sum_purify_mmbo_proj_B_rw += purify_mmbo_proj_B_rw
    sum_inverse_purify_mmbo_proj_B_rw += inverse_purify_mmbo_proj_B_rw
    sum_NMI_mmbo_proj_B_rw += NMI_mmbo_proj_B_rw



    # MMBO using finite difference L_sym
    start_time_MMBO_using_finite_difference_sym = time.time()
    u_MMBO_using_finite_difference_sym, num_iteration_MMBO_using_finite_difference_sym, MMBO_using_finite_difference_sym_modularity_list = MMBO_using_finite_differendce(m,degree_W_MMBO, 
                                        E_mmbo_sym, V_mmbo_sym, modularity_tol, N_t,  u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
    time_MMBO_using_finite_difference_sym = time.time() - start_time_MMBO_using_finite_difference_sym
    time_MMBO_using_finite_difference_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_using_finite_difference_sym
    #print('the number of MBO iteration for MMBO using finite difference with L_W&P_sym: ',num_iteration_MMBO_using_finite_difference_sym)

    u_MMBO_using_finite_difference_sym_label = vector_to_labels(u_MMBO_using_finite_difference_sym)
    u_MMBO_using_finite_difference_sym_dict = label_to_dict (u_MMBO_using_finite_difference_sym_label)
    u_MMBO_using_finite_difference_sym_list = dict_to_list_set(u_MMBO_using_finite_difference_sym_dict)
    u_MMBO_using_finite_difference_sym_coms = NodeClustering(u_MMBO_using_finite_difference_sym_list, graph=None)
    
    ER_modularity_MMBO_using_finite_difference_sym = evaluation.erdos_renyi_modularity(G, u_MMBO_using_finite_difference_sym_coms)[2]
    #modularity_MMBO_using_finite_difference_sym = evaluation.newman_girvan_modularity(G, u_MMBO_using_finite_difference_sym_coms)[2]
    #modularity_MMBO_using_finite_difference_sym = skn.clustering.modularity(W_MMBO,u_MMBO_using_finite_difference_sym,resolution=gamma)
    ARI_MMBO_using_finite_difference_sym = adjusted_rand_score(u_MMBO_using_finite_difference_sym_label, gt_labels_MMBO)
    purify_MMBO_using_finite_difference_sym = purity_score(gt_labels_MMBO, u_MMBO_using_finite_difference_sym_label)
    inverse_purify_MMBO_using_finite_difference_sym1 = inverse_purity_score(gt_labels_MMBO, u_MMBO_using_finite_difference_sym_label)
    NMI_MMBO_using_finite_difference_sym = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_using_finite_difference_sym_label)
    
    #print('modularity for MMBO using finite difference with L_W&P: ', modularity_MMBO_using_finite_difference_sym)
    #print('ARI for MMBO using finite difference with L_W&P: ', ARI_MMBO_using_finite_difference_sym)
    #print('purify for MMBO using finite difference with L_W&P: ', purify_MMBO_using_finite_difference_sym)
    #print('inverse purify for MMBO using finite difference with L_W&P: ', inverse_purify_MMBO_using_finite_difference_sym1)
    #print('NMI for MMBO using finite difference with L_W&P: ', NMI_MMBO_using_finite_difference_sym)

    sum_time_MMBO_using_finite_difference_sym += time_MMBO_using_finite_difference_sym
    sum_num_iteration_MMBO_using_finite_difference_sym += num_iteration_MMBO_using_finite_difference_sym 
    #sum_modularity_MMBO_using_finite_difference_sym += modularity_MMBO_using_finite_difference_sym
    sum_ER_modularity_MMBO_using_finite_difference_sym += ER_modularity_MMBO_using_finite_difference_sym
    sum_ARI_MMBO_using_finite_difference_sym += ARI_MMBO_using_finite_difference_sym
    sum_purify_MMBO_using_finite_difference_sym += purify_MMBO_using_finite_difference_sym
    sum_inverse_purify_MMBO_using_finite_difference_sym += inverse_purify_MMBO_using_finite_difference_sym1
    sum_NMI_MMBO_using_finite_difference_sym += NMI_MMBO_using_finite_difference_sym


    # MMBO using finite difference L_rw
    start_time_MMBO_using_finite_difference_rw = time.time()
    u_MMBO_using_finite_difference_rw, num_iteration_MMBO_using_finite_difference_rw, MMBO_using_finite_difference_rw_modularity_list = MMBO_using_finite_differendce(m,degree_W_MMBO, 
                                        E_mmbo_rw, V_mmbo_rw, modularity_tol, N_t,  u_init, W_MMBO, gamma=gamma, stopping_condition='modularity')
    time_MMBO_using_finite_difference_rw = time.time() - start_time_MMBO_using_finite_difference_rw
    time_MMBO_using_finite_difference_rw = time_eig_l_mix_rw + time_initialize_u + time_MMBO_using_finite_difference_rw
    #print('the number of MBO iteration for MMBO using inner step with L_W&P_rw: ',num_repeat_inner_rw)

    u_MMBO_using_finite_difference_rw_label = vector_to_labels(u_MMBO_using_finite_difference_rw)
    u_MMBO_using_finite_difference_rw_dict = label_to_dict(u_MMBO_using_finite_difference_rw_label)
    u_MMBO_using_finite_difference_rw_list = dict_to_list_set(u_MMBO_using_finite_difference_rw_dict)
    u_MMBO_using_finite_difference_rw_coms = NodeClustering(u_MMBO_using_finite_difference_rw_list, graph=None)
    
    ER_modularity_MMBO_using_finite_difference_rw = evaluation.erdos_renyi_modularity(G, u_MMBO_using_finite_difference_rw_coms)[2]
    #modularity_MMBO_using_finite_difference_rw = evaluation.newman_girvan_modularity(G, u_MMBO_using_finite_difference_rw_coms)[2]
    #modularity_MMBO_using_finite_difference_rw = skn.clustering.modularity(W_MMBO,u_MMBO_using_finite_difference_rw,resolution=gamma)
    ARI_MMBO_using_finite_difference_rw = adjusted_rand_score(u_MMBO_using_finite_difference_rw_label, gt_labels_MMBO)
    purify_MMBO_using_finite_difference_rw = purity_score(gt_labels_MMBO, u_MMBO_using_finite_difference_rw_label)
    inverse_purify_MMBO_using_finite_difference_rw = inverse_purity_score(gt_labels_MMBO, u_MMBO_using_finite_difference_rw_label)
    NMI_MMBO_using_finite_difference_rw = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_using_finite_difference_rw_label)


    #print('modularity for MMBO using inner step with L_W&P_rw: ', modularity_MMBO_using_finite_difference_rw)
    #print('ARI for MMBO using inner step with L_W&P_rw: ', ARI_MMBO_using_finite_difference_rw)
    #print('purify for MMBO using inner step with L_W&P_rw: ', purify_MMBO_using_finite_difference_rw)
    #print('inverse purify for MMBO using inner step with L_W&P_rw: ', inverse_purify_MMBO_using_finite_difference_rw)
    #print('NMI for MMBO using inner step with L_W&P_rw: ', NMI_MMBO_using_finite_difference_rw)

    sum_time_MMBO_using_finite_difference_rw += time_MMBO_using_finite_difference_rw
    sum_num_iteration_MMBO_using_finite_difference_rw += num_iteration_MMBO_using_finite_difference_rw 
    #sum_modularity_MMBO_using_finite_difference_rw += modularity_MMBO_using_finite_difference_rw
    sum_ER_modularity_MMBO_using_finite_difference_rw += ER_modularity_MMBO_using_finite_difference_rw
    sum_ARI_MMBO_using_finite_difference_rw += ARI_MMBO_using_finite_difference_rw
    sum_purify_MMBO_using_finite_difference_rw += purify_MMBO_using_finite_difference_rw
    sum_inverse_purify_MMBO_using_finite_difference_rw += inverse_purify_MMBO_using_finite_difference_rw
    sum_NMI_MMBO_using_finite_difference_rw += NMI_MMBO_using_finite_difference_rw


    # MMBO using finite difference B_sym
    start_time_MMBO_using_finite_difference_B_sym = time.time()
    u_MMBO_using_finite_difference_B_sym, num_iteration_MMBO_using_finite_difference_B_sym, MMBO_using_finite_difference_B_sym_modularity_list = MMBO_using_finite_differendce(m,degree_W_B, 
                                        D_mmbo_B_sym, V_mmbo_B_sym, modularity_tol, N_t,  u_init, W_B, gamma=gamma, stopping_condition='modularity')
    time_start_time_MMBO_using_finite_difference_B_sym = time.time() - start_time_MMBO_using_finite_difference_B_sym
    time_start_time_MMBO_using_finite_difference_B_sym = time_eig_B_sym + time_initialize_u + time_start_time_MMBO_using_finite_difference_B_sym
    #print('the number of MBO iteration for MMBO using inner step with L_B_sym: ',num_repeat_inner_nor_B_sym)

    u_MMBO_using_finite_difference_B_sym_label = vector_to_labels(u_MMBO_using_finite_difference_B_sym)
    u_MMBO_using_finite_difference_B_sym_dict = label_to_dict(u_MMBO_using_finite_difference_B_sym_label)
    u_MMBO_using_finite_difference_B_sym_list = dict_to_list_set(u_MMBO_using_finite_difference_B_sym_dict)
    u_MMBO_using_finite_difference_B_sym_coms = NodeClustering(u_MMBO_using_finite_difference_B_sym_list, graph=None)
    
    ER_modularity_MMBO_using_finite_difference_B_sym = evaluation.erdos_renyi_modularity(G, u_MMBO_using_finite_difference_B_sym_coms)[2]
    #modularity_MMBO_using_finite_difference_B_sym = evaluation.newman_girvan_modularity(G, u_MMBO_using_finite_difference_B_sym_coms)[2]
    #modularity_MMBO_using_finite_difference_B_sym = skn.clustering.modularity(W_B,u_MMBO_using_finite_difference_B_sym,resolution=gamma)
    ARI_MMBO_using_finite_difference_B_sym = adjusted_rand_score(u_MMBO_using_finite_difference_B_sym_label, gt_labels_B)
    purify_MMBO_using_finite_difference_B_sym = purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)
    inverse_purify_MMBO_using_finite_difference_B_sym = inverse_purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)
    NMI_MMBO_using_finite_difference_B_sym = normalized_mutual_info_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)

    sum_MMBO_using_finite_difference_B_sym += time_start_time_MMBO_using_finite_difference_B_sym
    sum_num_repeat_inner_nor_B_sym += num_iteration_MMBO_using_finite_difference_B_sym 
    #sum_modularity_mmbo_inner_B_sym += modularity_MMBO_using_finite_difference_B_sym
    sum_ER_modularity_mmbo_inner_B_sym += ER_modularity_MMBO_using_finite_difference_B_sym
    sum_ARI_mmbo_inner_B_sym += ARI_MMBO_using_finite_difference_B_sym
    sum_purify_mmbo_inner_B_sym += purify_MMBO_using_finite_difference_B_sym
    sum_inverse_purify_mmbo_inner_B_sym += inverse_purify_MMBO_using_finite_difference_B_sym
    sum_NMI_mmbo_inner_B_sym += NMI_MMBO_using_finite_difference_B_sym


    # MMBO using finite difference B_rw
    start_time_MMBO_using_finite_difference_B_rw = time.time()
    u_MMBO_using_finite_difference_B_rw, num_iertation_MMBO_using_finite_difference_B_rw, MMBO_using_finite_difference_B_rw_modularity_list = MMBO_using_finite_differendce(m,degree_W_B, 
                                        D_mmbo_B_rw, V_mmbo_B_rw, modularity_tol, N_t,  u_init, W_B, gamma=gamma, stopping_condition='modularity')
    time_MMBO_using_finite_difference_B_rw = time.time() - start_time_MMBO_using_finite_difference_B_rw
    #print('the number of MBO iteration for MMBO using inner step with L_B_rw: ',num_repeat_inner_B_rw)

    u_MMBO_using_finite_difference_B_rw_label = vector_to_labels(u_MMBO_using_finite_difference_B_rw)
    u_MMBO_using_finite_difference_B_rw_dict = label_to_dict(u_MMBO_using_finite_difference_B_rw_label)
    u_MMBO_using_finite_difference_B_rw_list = dict_to_list_set(u_MMBO_using_finite_difference_B_rw_dict)
    u_MMBO_using_finite_difference_B_rw_coms = NodeClustering(u_MMBO_using_finite_difference_B_rw_list, graph=None)
    
    ER_modularity_MMBO_using_finite_difference_B_rw = evaluation.erdos_renyi_modularity(G, u_MMBO_using_finite_difference_B_rw_coms)[2]
    #modularity_MMBO_using_finite_difference_B_rw = evaluation.newman_girvan_modularity(G, u_MMBO_using_finite_difference_B_rw_coms)[2]
    #modularity_MMBO_using_finite_difference_B_rw = skn.clustering.modularity(W_B,u_MMBO_using_finite_difference_B_rw,resolution=1)
    ARI_mmbo_inner_B_rwMMBO_using_finite_difference_B_rw = adjusted_rand_score(u_MMBO_using_finite_difference_B_rw_label, gt_labels_B)
    purify_MMBO_using_finite_difference_B_rw = purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)
    inverse_purifyMMBO_using_finite_difference_B_rw = inverse_purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)
    NMI_MMBO_using_finite_difference_B_rw = normalized_mutual_info_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)


    sum_time_MMBO_using_finite_difference_B_rw += time_MMBO_using_finite_difference_B_rw
    sum_num_iertation_MMBO_using_finite_difference_B_rw += num_iertation_MMBO_using_finite_difference_B_rw 
    #sum_modularity_mmbo_inner_B_rw += modularity_MMBO_using_finite_difference_B_rw
    sum_ER_modularity_mmbo_inner_B_rw += ER_modularity_MMBO_using_finite_difference_B_rw
    sum_ARI_mmbo_inner_B_rw += ARI_mmbo_inner_B_rwMMBO_using_finite_difference_B_rw
    sum_purify_mmbo_inner_B_rw += purify_MMBO_using_finite_difference_B_rw
    sum_inverse_purify_mmbo_inner_B_rw += inverse_purifyMMBO_using_finite_difference_B_rw
    sum_NMI_mmbo_inner_B_rw += NMI_MMBO_using_finite_difference_B_rw


print('MMBO using projection L_sym')
average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 20
average_num_iter_MMBO_projection_sym = sum_num_iteration_MMBO_projection_l_sym / 20
#average_modularity_MMBO_projection_sym = sum_modularity_MMBO_projection_l_sym / 20
average_ER_modularity_MMBO_projection_sym = sum_ER_modularity_MMBO_projection_l_sym / 20
average_ARI_MMBO_projection_sym = sum_ARI_MMBO_projection_l_sym / 20
average_purify_MMBO_projection_sym = sum_purify_MMBO_projection_l_sym / 20
average_inverse_purify_MMBO_projection_sym = sum_inverse_purify_MMBO_projection_l_sym / 20
average_NMI_MMBO_projection_sym = sum_NMI_MMBO_projection_l_sym / 20


print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_sym)
#print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_sym)
print('average_ER_modularity_MMBO_projection_sym: ', average_ER_modularity_MMBO_projection_sym)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_sym)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_sym)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_sym)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_sym)


print('MMBO using projection L_rw')
average_time_MMBO_projection_rw = sum_time_MMBO_projection_rw / 20
average_num_iteration_MMBO_projection_rw = sum_num_iteration_MMBO_projection_l_rw / 20
#average_modularity_MMBO_projection_rw = sum_modularity_MMBO_projection_l_rw / 20
average_ER_modularity_MMBO_projection_rw = sum_ER_modularity_MMBO_projection_l_rw / 20
average_ARI_MMBO_projection_rw = sum_ARI_MMBO_projection_l_rw / 20
average_purify_MMBO_projection_rw = sum_purify_MMBO_projection_l_rw / 20
average_inverse_purify_MMBO_projection_rw = sum_inverse_purify_MMBO_projection_l_rw / 20
average_NMI_MMBO_projection_rw = sum_NMI_MMBO_projection_l_rw / 20


print('average_time_MMBO_projection_rw: ', average_time_MMBO_projection_rw)
print('average_num_iteration_MMBO_projection_rw: ', average_num_iteration_MMBO_projection_rw)
#print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_rw)
print('average_ER_modularity_MMBO_projection_sym: ', average_ER_modularity_MMBO_projection_rw)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_rw)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_rw)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_rw)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_rw)


print('MMBO using projection B_sym')
average_time_MMBO_projection_B_sym = sum_time_MMBO_projection_B_sym / 20
average_num_iter_MMBO_projection_B_sym = sum_num_repeat_mmbo_proj_B_sym / 20
#average_modularity_MMBO_projection_B_sym = sum_modularity_mmbo_proj_B_sym / 20
average_ER_modularity_MMBO_projection_B_sym = sum_ER_modularity_mmbo_proj_B_sym / 20
average_ARI_MMBO_projection_B_sym = sum_ARI_mmbo_proj_B_sym / 20
average_purify_MMBO_projection_B_sym = sum_purify_mmbo_proj_B_sym / 20
average_inverse_purify_MMBO_projection_B_sym = sum_inverse_purify_mmbo_proj_B_sym / 20
average_NMI_MMBO_projection_B_sym = sum_NMI_mmbo_proj_B_sym / 20


print('average_time_MMBO_projection_B_sym: ', average_time_MMBO_projection_B_sym)
print('average_num_iteration_MMBO_projection_B_sym: ', average_num_iter_MMBO_projection_B_sym)
#print('average_modularity_MMBO_projection_B_sym: ', average_modularity_MMBO_projection_B_sym)
print('average_ER_modularity_MMBO_projection_B_sym: ', average_ER_modularity_MMBO_projection_B_sym)
print('average_ARI_MMBO_projection_B_sym: ', average_ARI_MMBO_projection_B_sym)
print('average_purify_MMBO_projection_B_sym: ', average_purify_MMBO_projection_B_sym)
print('average_inverse_purify_MMBO_projection_B_sym: ', average_inverse_purify_MMBO_projection_B_sym)
print('average_NMI_MMBO_projection_B_sym: ', average_NMI_MMBO_projection_B_sym)


print('MMBO using projection B_rw')
average_time_MMBO_projection_B_rw = sum_time_MMBO_projection_B_rw / 20
average_num_iter_MMBO_projection_B_rw = sum_num_iteration_mmbo_proj_B_rw / 20
#average_modularity_MMBO_projection_B_rw = sum_modularity_mmbo_proj_B_rw / 20
average_ER_modularity_MMBO_projection_B_rw = sum_ER_modularity_mmbo_proj_B_rw / 20
average_ARI_MMBO_projection_B_rw = sum_ARI_mmbo_proj_B_rw / 20
average_purify_MMBO_projection_B_rw = sum_purify_mmbo_proj_B_rw / 20
average_inverse_purify_MMBO_projection_B_rw = sum_inverse_purify_mmbo_proj_B_rw / 20
average_NMI_MMBO_projection_B_rw = sum_NMI_mmbo_proj_B_rw / 20


print('average_time_MMBO_projection_B_rw: ', average_time_MMBO_projection_B_rw)
print('average_num_iteration_MMBO_projection_B_rw: ', average_num_iter_MMBO_projection_B_rw)
#print('average_modularity_MMBO_projection_B_rw: ', average_modularity_MMBO_projection_B_rw)
print('average_ER_modularity_MMBO_projection_B_rw: ', average_ER_modularity_MMBO_projection_B_rw)
print('average_ARI_MMBO_projection_B_rw: ', average_ARI_MMBO_projection_B_rw)
print('average_purify_MMBO_projection_symMMBO_projection_B_rw: ', average_purify_MMBO_projection_B_rw)
print('average_inverse_purify_MMBO_projection_B_rw: ', average_inverse_purify_MMBO_projection_B_rw)
print('average_NMI_MMBO_projection_B_rw: ', average_NMI_MMBO_projection_B_rw)


print('MMBO using finite difference L_sym')
average_time_MMBO_inner_step = sum_time_MMBO_using_finite_difference_sym / 20
average_num_iter_MMBO_inner_step = sum_num_iteration_MMBO_using_finite_difference_sym / 20
#average_modularity_MMBO_inner_step = sum_modularity_MMBO_using_finite_difference_sym / 20
average_ER_modularity_MMBO_inner_step = sum_ER_modularity_MMBO_using_finite_difference_sym / 20
average_ARI_MMBO_inner_step = sum_ARI_MMBO_using_finite_difference_sym / 20
average_purify_MMBO_inner_step = sum_purify_MMBO_using_finite_difference_sym / 20
average_inverse_purify_MMBO_inner_step = sum_inverse_purify_MMBO_using_finite_difference_sym / 20
average_NMI_MMBO_inner_step = sum_NMI_MMBO_using_finite_difference_sym / 20


print('average_time_MMBO_using_finite_difference_sym: ', average_time_MMBO_inner_step)
print('average_num_iteration_MMBO_using_finite_difference_sym: ', average_num_iter_MMBO_inner_step)
#print('average_modularity_MMBO_using_finite_difference_sym: ', average_modularity_MMBO_inner_step)
print('average_ER_modularity_MMBO_using_finite_difference_sym: ', average_ER_modularity_MMBO_inner_step)
print('average_ARI_MMBO_using_finite_difference_sym: ', average_ARI_MMBO_inner_step)
print('average_purify_MMBO_using_finite_difference_sym: ', average_purify_MMBO_inner_step)
print('average_inverse_purify_MMBO_using_finite_difference_sym: ', average_inverse_purify_MMBO_inner_step)
print('average_NMI_MMBO_using_finite_difference_sym: ', average_NMI_MMBO_inner_step)


print('MMBO using finite difference L_rw')
average_time_MMBO_using_finite_difference_rw = sum_time_MMBO_using_finite_difference_rw / 20
average_num_iter_MMBO_inner_step_rw = sum_num_iteration_MMBO_using_finite_difference_rw / 20
#average_modularity_MMBO_inner_step_rw = sum_modularity_MMBO_using_finite_difference_rw / 20
average_ER_modularity_MMBO_inner_step_rw = sum_ER_modularity_MMBO_using_finite_difference_rw / 20
average_ARI_MMBO_inner_step_rw = sum_ARI_MMBO_using_finite_difference_rw / 20
average_purify_MMBO_inner_step_rw = sum_purify_MMBO_using_finite_difference_rw / 20
average_inverse_purify_MMBO_inner_step_rw = sum_inverse_purify_MMBO_using_finite_difference_rw / 20
average_NMI_MMBO_inner_step_rw = sum_NMI_MMBO_using_finite_difference_rw / 20


print('average_time_MMBO_using_finite_difference_rw: ', average_time_MMBO_using_finite_difference_rw)
print('average_num_iter_MMBO_using_finite_difference_rw: ', average_num_iter_MMBO_inner_step_rw)
#print('average_modularity_MMBO_using_finite_difference_rw: ', average_modularity_MMBO_inner_step_rw)
print('average_ER_modularity_MMBO_using_finite_difference_rw: ', average_ER_modularity_MMBO_inner_step_rw)
print('average_ARI_MMBO_using_finite_difference_rw: ', average_ARI_MMBO_inner_step_rw)
print('average_purify_MMBO_using_finite_difference_rw: ', average_purify_MMBO_inner_step_rw)
print('average_inverse_purify_MMBO_using_finite_difference_rw: ', average_inverse_purify_MMBO_inner_step_rw)
print('average_NMI_MMBO_using_finite_difference_rw: ', average_NMI_MMBO_inner_step_rw)


print('MMBO using finite difference B_sym')
average_time_MMBO_using_finite_difference_B_sym = sum_MMBO_using_finite_difference_B_sym / 20
average_num_iter_MMBO_inner_step_B_sym = sum_num_repeat_inner_nor_B_sym / 20
#average_modularity_MMBO_inner_step_B_sym = sum_modularity_mmbo_inner_B_sym / 20
average_ER_modularity_MMBO_inner_step_B_sym = sum_ER_modularity_mmbo_inner_B_sym / 20
average_ARI_MMBO_inner_step_B_sym = sum_ARI_mmbo_inner_B_sym / 20
average_purify_MMBO_inner_step_B_sym = sum_purify_mmbo_inner_B_sym / 20
average_inverse_purify_MMBO_inner_step_B_sym = sum_inverse_purify_mmbo_inner_B_sym / 20
average_NMI_MMBO_inner_step_B_sym = sum_NMI_mmbo_inner_B_sym / 20


print('average_time_MMBO_using_finite_difference_B_sym: ',average_time_MMBO_using_finite_difference_B_sym)
print('average_num_iteration_MMBO_using_finite_difference_B_sym: ', average_num_iter_MMBO_inner_step_B_sym)
#print('average_modularity_MMBO_using_finite_difference_B_sym: ', average_modularity_MMBO_inner_step_B_sym)
print('average_ER_modularity_MMBO_using_finite_difference_B_sym: ', average_ER_modularity_MMBO_inner_step_B_sym)
print('average_ARI_MMBO_using_finite_difference_B_sym: ', average_ARI_MMBO_inner_step_B_sym)
print('average_purify_MMBO_using_finite_difference_B_symp: ', average_purify_MMBO_inner_step_B_sym)
print('average_inverse_purify_MMBO_using_finite_difference_B_sym: ', average_inverse_purify_MMBO_inner_step_B_sym)
print('average_NMI_MMBO_using_finite_difference_B_sym: ', average_NMI_MMBO_inner_step_B_sym)


print('MMBO using finite difference B_rw')
average_time_MMBO_using_finite_difference_B_rw = sum_time_MMBO_using_finite_difference_B_rw /20
average_num_iter_MMBO_inner_step_B_rw = sum_num_iertation_MMBO_using_finite_difference_B_rw / 20
#average_modularity_MMBO_inner_step_B_rw = sum_modularity_mmbo_inner_B_rw / 20
average_ER_modularity_MMBO_inner_step_B_rw = sum_ER_modularity_mmbo_inner_B_rw / 20
average_ARI_MMBO_inner_step_B_rw = sum_ARI_mmbo_inner_B_rw / 20
average_purify_MMBO_inner_step_B_rw = sum_purify_mmbo_inner_B_rw / 20
average_inverse_purify_MMBO_inner_step_B_rw = sum_inverse_purify_mmbo_inner_B_rw / 20
average_NMI_MMBO_inner_step_B_rw = sum_NMI_mmbo_inner_B_rw / 20


print('average_time_MMBO_using_finite_difference_B_rw: ',average_time_MMBO_using_finite_difference_B_rw)
print('average_num_ieration_MMBO_using_finite_difference_B_rw: ', average_num_iter_MMBO_inner_step_B_rw)
#print('average_modularity_MMBO_using_finite_difference_B_rw: ', average_modularity_MMBO_inner_step_B_rw)
print('average_ER_modularity_MMBO_using_finite_difference_B_rw: ', average_ER_modularity_MMBO_inner_step_B_rw)
print('average_ARI_MMBO_using_finite_difference_B_rw: ', average_ARI_MMBO_inner_step_B_rw)
print('average_purify_MMBO_using_finite_difference_B_rw: ', average_purify_MMBO_inner_step_B_rw)
print('average_inverse_purify_MMBO_using_finite_difference_B_rw: ', average_inverse_purify_MMBO_inner_step_B_rw)
print('average_NMI_MMBO_using_finite_difference_B_rw: ', average_NMI_MMBO_inner_step_B_rw)



print('HU method L_sym')
average_time_hu_mbo = sum_time_hu_sym / 20
average_num_iter_HU_sym = sum_num_iteration_HU_sym / 20
#average_modularity_hu_sym = sum_modularity_hu_sym / 20
average_ER_modularity_hu_sym = sum_ER_modularity_hu_sym / 20
average_ARI_hu_original_sym = sum_ARI_hu_original_sym / 20
average_purify_hu_original_sym = sum_purify_hu_original_sym / 20
average_inverse_purify_hu_original_sym = sum_inverse_purify_hu_original_sym / 20
average_NMI_hu_original_sym = sum_NMI_hu_original_sym / 20


print('average_time_HU_sym: ', average_time_hu_mbo)
print('average_num_iteration_HU_sym: ', average_num_iter_HU_sym)
#print('average_modularity_HU_sym: ', average_modularity_hu_sym)
print('average_ER_modularity_HU_sym: ', average_ER_modularity_hu_sym)
print('average_ARI_HU_sym: ', average_ARI_hu_original_sym)
print('average_purify_HU_sym: ', average_purify_hu_original_sym)
print('average_inverse_purify_HU_sym: ', average_inverse_purify_hu_original_sym)
print('average_NMI_HU_sym: ', average_NMI_hu_original_sym)


print('HU method L_rw')
average_time_hu_mbo = sum_time_hu_rw / 20
average_num_iter_HU_rw = sum_num_iter_HU_rw / 20
#average_modularity_hu_rw = sum_modularity_hu_rw / 20
average_ER_modularity_hu_rw = sum_ER_modularity_hu_rw / 20
average_ARI_hu_original_rw = sum_ARI_hu_original_rw / 20
average_purify_hu_original_rw = sum_purify_hu_original_rw / 20
average_inverse_purify_hu_original_rw = sum_inverse_purify_hu_original_rw / 20
average_NMI_hu_original_rw = sum_NMI_hu_original_rw / 20



print('average_time_HU_rw: ', average_time_hu_mbo)
print('average_num_iteration_HU_rw: ', average_num_iter_HU_rw)
#print('average_modularity_HU_rw: ', average_modularity_hu_rw)
print('average_ER_modularity_HU_rw: ', average_ER_modularity_hu_rw)
print('average_ARI_HU_rw: ', average_ARI_hu_original_rw)
print('average_purify_HU_rw: ', average_purify_hu_original_rw)
print('average_inverse_purify_HU_rw: ', average_inverse_purify_hu_original_rw)
print('average_NMI_HU_rw: ', average_NMI_hu_original_rw)






# Spectral clustering with k-means
sum_time_sc=0
sum_modularity_sc =0
sum_ER_modularity_sc =0
sum_ARI_spectral_clustering = 0
sum_purify_spectral_clustering = 0
sum_inverse_purify_spectral_clustering = 0
sum_NMI_spectral_clustering = 0

for _ in range(5):
    start_time_spectral_clustering = time.time()
    sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed')
    assignment = sc.fit_predict(W)
    time_sc = time.time() - start_time_spectral_clustering
    #print("spectral clustering algorithm:-- %.3f seconds --" % (time_sc))

    ass_vec = labels_to_vector(assignment)
    ass_dict = label_to_dict(assignment)
    ass_list = dict_to_list_set(ass_dict)
    ass_coms = NodeClustering(ass_list, graph=None)
    
    ER_modularity_spectral_clustering = evaluation.erdos_renyi_modularity(G, ass_coms)[2]
    #modularity_spectral_clustering = evaluation.newman_girvan_modularity(G, ass_coms)[2]
    #modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=1)
    ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)
    purify_spectral_clustering = purity_score(gt_labels, assignment)
    inverse_purify_spectral_clustering = inverse_purity_score(gt_labels, assignment)
    NMI_spectral_clustering = normalized_mutual_info_score(gt_labels, assignment)

    #print('modularity Spectral clustering score: ', modularity_spectral_clustering)
    #print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
    #print('purify for Spectral clustering : ', purify_spectral_clustering)
    #print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
    #print('NMI for Spectral clustering: ', NMI_spectral_clustering)
    
    sum_time_sc += time_sc
    #sum_modularity_sc += modularity_spectral_clustering
    sum_ER_modularity_sc += ER_modularity_spectral_clustering
    sum_ARI_spectral_clustering += ARI_spectral_clustering
    sum_purify_spectral_clustering += purify_spectral_clustering
    sum_inverse_purify_spectral_clustering += inverse_purify_spectral_clustering
    sum_NMI_spectral_clustering += NMI_spectral_clustering

average_time_sc = sum_time_sc / 5
#average_modularity_sc = sum_modularity_sc / 5
average_ER_modularity_sc = sum_ER_modularity_sc / 5
average_ARI_spectral_clustering = sum_ARI_spectral_clustering / 5
average_purify_spectral_clustering = sum_purify_spectral_clustering / 5
average_inverse_purify_spectral_clustering = sum_inverse_purify_spectral_clustering / 5
average_NMI_spectral_clustering = sum_NMI_spectral_clustering / 5

print('Spectral clustering')
print('average_time_sc: ', average_time_sc)
#print('average_modularity_sc: ', average_modularity_sc)
print('average_ER_modularity_sc: ', average_ER_modularity_sc)
print('average_ARI_spectral_clustering: ', average_ARI_spectral_clustering)
print('average_purify_spectral_clustering: ', average_purify_spectral_clustering)
print('average_inverse_purify_spectral_clustering: ', average_inverse_purify_spectral_clustering)
print('average_NMI_spectral_clustering: ', average_NMI_spectral_clustering)



# CNM algorithm (can setting resolution gamma)
sum_time_CNM =0
sum_modularity_CNM =0
sum_ER_modularity_CNM =0
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
    CNM_coms = NodeClustering(partition_CNM_list, graph=None)
    
    ER_modularity_CNM = evaluation.erdos_renyi_modularity(G, CNM_coms)[2]
    #modularity_CNM = evaluation.newman_girvan_modularity(G, CNM_coms)[2]
    #modularity_CNM = skn.clustering.modularity(W,CNM_array_sorted,resolution=gamma)
    ARI_CNM = adjusted_rand_score(CNM_array_sorted, gt_labels)
    purify_CNM = purity_score(gt_labels, CNM_array_sorted)
    inverse_purify_CNM = inverse_purity_score(gt_labels, CNM_array_sorted)
    NMI_CNM = normalized_mutual_info_score(gt_labels, CNM_array_sorted)

    sum_time_CNM += time_CNM
    #sum_modularity_CNM += modularity_CNM
    sum_ER_modularity_CNM += ER_modularity_CNM
    sum_ARI_CNM += ARI_CNM
    sum_purity_CNM += purify_CNM
    sum_inverse_purity_CNM += inverse_purify_CNM
    sum_NMI_CNM += NMI_CNM


average_time_CNM = sum_time_CNM / 20
#average_modularity_CNM = sum_modularity_CNM / 20
average_ER_modularity_CNM = sum_ER_modularity_CNM / 20
average_ARI_CNM = sum_ARI_CNM / 20
average_purity_CNM = sum_purity_CNM / 20
average_inverse_purity_CNM = sum_inverse_purity_CNM / 20
average_NMI_CNM = sum_NMI_CNM / 20


print('CNM')
print('average_time_CNM: ', average_time_CNM)
#print('average_modularity_CNM: ', average_modularity_CNM)
print('average_ER_modularity_CNM: ', average_ER_modularity_CNM)
print('average_ARI_CNM: ', average_ARI_CNM)
print('average_purity_CNM: ', average_purity_CNM)
print('average_inverse_purity_CNM: ', average_inverse_purity_CNM)
print('average_NMI_CNM: ', average_NMI_CNM)


