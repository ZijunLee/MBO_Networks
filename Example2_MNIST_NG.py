from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import time
import utils
import random
from community import community_louvain
from Nystrom_extension_QR import nystrom_QR_l_sym, nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
import graphlearning as gl
from MMBO_HU_Boyd import MMBO_using_projection, MMBO_using_finite_differendce,HU_mmbo_method, boyd_mbo_method
from utils import vector_to_labels, labels_to_vector, purity_score, inverse_purity_score, generate_initial_value_multiclass




# Example 2: MNIST

# Parameter setting

# num_communities is found by Louvain
# choose m = num_communities
tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 0.5
tau = 0.02
num_nodes = 70000
num_communities = 250

# Load MNIST data, ground truth, and build 10-nearest neighbor weight matrix
data, gt_labels = gl.datasets.load('mnist')
#gt_vec = labels_to_vector(gt_labels)
print('gt_labels', gt_labels.shape)

pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)
W = gl.weightmatrix.knn(Z_training, 10, symmetrize=True)



# First run the Louvain method in order to get the number of clusters
sum_time_louvain=0
sum_modularity_louvain =0
sum_ARI_louvain = 0
sum_purity_louvain = 0
sum_inverse_purity_louvain = 0
sum_NMI_louvain = 0
sum_louvain_cluster =0

G = nx.convert_matrix.from_scipy_sparse_matrix(W)

#for _ in range(20):
#    start_time_louvain = time.time()
#    partition_Louvain = community_louvain.best_partition(G, resolution=gamma, randomize=True)
#    time_louvain = time.time() - start_time_louvain
    #print("Louvain:-- %.3f seconds --" % (time_louvain))

#    louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#    louvain_array_sorted = np.asarray(louvain_list)

#    louvain_vec = labels_to_vector(louvain_array_sorted)

#    louvain_cluster = len(np.unique(louvain_array_sorted))

#    modularity_louvain = skn.clustering.modularity(W,louvain_array_sorted,resolution=gamma)
#   ARI_louvain = adjusted_rand_score(louvain_array_sorted, gt_labels)
#    purify_louvain = purity_score(gt_labels, louvain_array_sorted)
#    inverse_purify_louvain = inverse_purity_score(gt_labels, louvain_array_sorted)
#    NMI_louvain = normalized_mutual_info_score(gt_labels, louvain_array_sorted)

#    sum_louvain_cluster += louvain_cluster
#    sum_time_louvain += time_louvain
#    sum_modularity_louvain += modularity_louvain
#    sum_ARI_louvain += ARI_louvain
#    sum_purity_louvain += purify_louvain
#    sum_inverse_purity_louvain += inverse_purify_louvain
#    sum_NMI_louvain += NMI_louvain

#average_louvain_cluster = sum_louvain_cluster / 20
#average_time_louvain = sum_time_louvain / 20
#average_modularity_louvain = sum_modularity_louvain / 20
#average_ARI_louvain = sum_ARI_louvain / 20
#average_purify_louvain = sum_purity_louvain / 20
#average_inverse_purify_louvain = sum_inverse_purity_louvain / 20
#average_NMI_louvain = sum_NMI_louvain / 20

#print('average_time_louvain: ', average_time_louvain)
#print('average_modularity_louvain: ', average_modularity_louvain)
#print('average_ARI_louvain: ', average_ARI_louvain)
#print('average_purify_louvain: ', average_purify_louvain)
#print('average_inverse_purify_louvain: ', average_inverse_purify_louvain)
#print('average_NMI_louvain: ', average_NMI_louvain)

#num_communities  = round(average_louvain_cluster)
m = num_communities
five_cluster = 5*num_communities


# Compute the eigenvalues and eigenvectors of L_sym and L_rw
eigenvalues_sym, eigenvectors_sym, rw_left_eigvec, order_raw_data_HU, index_HU, time_eig_l_sym, time_eig_l_rw = nystrom_QR_l_sym(Z_training, num_nystrom=five_cluster, tau=tau)
eig_val_HU_sym = np.squeeze(eigenvalues_sym[:m])
eig_vec_HU_sym = eigenvectors_sym[:,:m]
eig_vec_HU_rw = rw_left_eigvec[:,:m]


gt_labels_HU = gt_labels[index_HU]
gt_HU_vec = labels_to_vector(gt_labels_HU)
W_HU = gl.weightmatrix.knn(order_raw_data_HU, 10, kernel='gaussian')
degree_W_HU = np.array(np.sum(W_HU, axis=-1)).flatten()


eig_val_boyd_sym = np.squeeze(eigenvalues_sym[:five_cluster])
eig_vec_boyd_sym = eigenvectors_sym[:,:five_cluster]
eig_vec_boyd_rw = rw_left_eigvec[:,:five_cluster]

gt_labels_boyd = gt_labels[index_HU]
gt_boyd_vec = labels_to_vector(gt_labels_HU)
W_boyd = gl.weightmatrix.knn(order_raw_data_HU, 10, kernel='gaussian')
degree_W_boyd = np.array(np.sum(W_boyd, axis=-1)).flatten()


# Compute the eigenvalues and eigenvectors of L_mix_sym and L_mix_rw
eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw(Z_training, num_nystrom=500, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]


gt_labels_MMBO = gt_labels[index_MMBO]
gt_MMBO_projection_vec = labels_to_vector(gt_labels_MMBO)
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10, kernel='gaussian')
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()



eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym, eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw, order_raw_data_B, index_B, time_eig_B_sym, time_eig_B_rw = nystrom_QR_l_mix_B_sym_rw(Z_training, num_nystrom=500, tau=tau)
D_mmbo_B_sym = np.squeeze(eig_val_mmbo_B_sym[:m])
V_mmbo_B_sym = eig_vec_mmbo_B_sym[:,:m]
D_mmbo_B_rw = np.squeeze(eig_val_mmbo_B_rw[:m])
V_mmbo_B_rw = eig_vec_mmbo_B_rw[:,:m]


gt_labels_B = gt_labels[index_B]
gt_B_vec = labels_to_vector(gt_labels_B)
W_B = gl.weightmatrix.knn(order_raw_data_B, 10, kernel='gaussian')
degree_W_B = np.array(np.sum(W_B, axis=-1)).flatten()


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

sum_time_boyd_sym = 0
sum_num_iteration_boyd_l_sym = 0 
sum_modularity_boyd_l_sym = 0
sum_ARI_boyd_l_sym = 0
sum_purify_boyd_l_sym = 0
sum_inverse_purify_boyd_l_sym = 0
sum_NMI_boyd_l_sym = 0


sum_time_boyd_rw = 0
sum_num_iteration_boyd_l_rw = 0
sum_modularity_boyd_l_rw = 0
sum_ARI_boyd_l_rw = 0
sum_purify_boyd_l_rw = 0
sum_inverse_purify_boyd_l_rw = 0
sum_NMI_boyd_l_rw = 0

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




# run the script 20 times using the modularity − related stopping condition
for _ in range(20):

    start_time_initialize = time.time()
    # Unsupervised
    #print('Unsupervised')
    u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=num_communities)


    # 10% supervised
    #print('NG modularity -- 10% supervised, K=10')
    #u_init_sup = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=10)

    #row_numbers = range(0, len(gt_labels))
    #Rs = random.sample(row_numbers, 7000)
    
    #u_init_HU_sup = u_init_sup
    #u_init_HU_sup[[Rs],:] = gt_HU_vec[[Rs],:]

    #u_init_LWP_sup = u_init_sup
    #u_init_LWP_sup[[Rs],:] = gt_MMBO_projection_vec[[Rs],:]

    #u_init_B_sup = u_init_sup
    #u_init_B_sup[[Rs],:] = gt_B_vec[[Rs],:]

    #u_init_HU_sup, indices, train_onehot = utils.initialization(num_nodes, 10, gt_labels_HU, num_per_class=700)
    #u_init_LWP_sup, indices_MMBO_projection, train_onehot_MMBO_proj = utils.initialization(num_nodes, 10, gt_labels_MMBO, num_per_class=700)
    #u_init_B_sup, indices_MMBO_fd, train_onehot_MMBO_fd = utils.initialization(num_nodes, 10, gt_labels_B, num_per_class=700)

    time_initialize_u = time.time() - start_time_initialize


    start_time_HU_sym = time.time()
    u_hu_sym_vector, num_iteration_HU_sym, HU_sym_modularity_list = HU_mmbo_method(num_nodes, degree_W_HU, eig_val_HU_sym, eig_vec_HU_sym,
                                 modularity_tol, N_t, u_init, W_HU, gamma=gamma, stopping_condition='modularity') 
    time_HU_sym = time.time() - start_time_HU_sym
    time_HU_sym = time_eig_l_sym + time_initialize_u + time_HU_sym
    #print('the num_iteration of HU method with L_sym: ', num_iteration_HU_sym)

    u_hu_sym_label = vector_to_labels(u_hu_sym_vector)

    modularity_hu_sym = skn.clustering.modularity(W_HU, u_hu_sym_label,resolution=gamma)
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
    sum_modularity_hu_sym += modularity_hu_sym
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

    u_hu_label_rw_label = vector_to_labels(u_hu_vector_rw)

    modu_Hu_rw = skn.clustering.modularity(W_HU,u_hu_label_rw_label,resolution=gamma)
    ARI_Hu_rw = adjusted_rand_score(u_hu_label_rw_label, gt_labels_HU)
    purify_Hu_rw = purity_score(gt_labels_HU, u_hu_label_rw_label)
    inverse_purify_Hu_rw = inverse_purity_score(gt_labels_HU, u_hu_label_rw_label)
    NMI_Hu_rw = normalized_mutual_info_score(gt_labels_HU, u_hu_label_rw_label)

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


    # Boyd method using l_sym
    start_time_boyd_l_sym = time.time()
    u_boyd_vector_l_sym, num_iteration_boyd_l_sym, boyd_sym_modularity_list = boyd_mbo_method(num_communities, degree_W_HU,  
                                        eig_val_boyd_sym, eig_vec_boyd_sym, modularity_tol,u_init, W_HU, gamma=gamma, stopping_condition='modularity')
    time_boyd_sym = time.time() - start_time_boyd_l_sym
    time_boyd_sym = time_eig_l_sym + time_initialize_u + time_boyd_sym
    #print('the number of MBO iteration for Boyd method with L_sym: ', num_iteration_boyd_l_sym)

    u_boyd_l_sym_label = vector_to_labels(u_boyd_vector_l_sym)
    modularity_boyd_l_sym = skn.clustering.modularity(W_boyd ,u_boyd_l_sym_label,resolution=gamma)
    ARI_boyd_l_sym = adjusted_rand_score(u_boyd_l_sym_label, gt_labels_boyd)
    purify_boyd_l_sym = purity_score(gt_labels_boyd, u_boyd_l_sym_label)
    inverse_purify_boyd_l_sym = inverse_purity_score(gt_labels_boyd, u_boyd_l_sym_label)
    NMI_boyd_l_sym = normalized_mutual_info_score(gt_labels_boyd, u_boyd_l_sym_label)

    #print('modularity for Boyd method with L_sym: ', modularity_boyd_l_sym)
    #print('ARI for Boyd method with L_sym: ', ARI_boyd_l_sym)
    #print('purify for Boyd method with L_sym: ', purify_boyd_l_sym)
    #print('inverse purify for Boyd method with L_sym: ', inverse_purify_boyd_l_sym)
    #print('NMI for Boyd method with L_sym: ', NMI_boyd_l_sym)


    sum_time_boyd_sym += time_boyd_sym
    sum_num_iteration_boyd_l_sym += num_iteration_boyd_l_sym 
    sum_modularity_boyd_l_sym += modularity_boyd_l_sym
    sum_ARI_boyd_l_sym += ARI_boyd_l_sym
    sum_purify_boyd_l_sym += purify_boyd_l_sym
    sum_inverse_purify_boyd_l_sym += inverse_purify_boyd_l_sym
    sum_NMI_boyd_l_sym += NMI_boyd_l_sym


    # Boyd method using l_rw
    start_time_boyd_l_rw = time.time()
    u_boyd_l_rw, num_iteration_boyd_l_rw, boyd_rw_modularity_list = boyd_mbo_method(num_communities, degree_W_HU,  
                                        eig_val_boyd_sym, eig_vec_boyd_rw, modularity_tol, u_init, W_HU, gamma=gamma, stopping_condition='modularity')
    time_boyd_rw = time.time() - start_time_boyd_l_rw
    time_boyd_rw = time_eig_l_rw + time_initialize_u + time_boyd_rw
    #print('the number of MBO iteration for Boyd method with L_rw: ', num_iteration_boyd_l_rw)

    u_boyd_l_rw_label = vector_to_labels(u_boyd_l_rw)
    modularity_boyd_l_rw = skn.clustering.modularity(W_boyd ,u_boyd_l_rw_label,resolution=gamma)
    ARI_boyd_l_rw = adjusted_rand_score(u_boyd_l_rw_label, gt_labels_boyd)
    purify_boyd_l_rw = purity_score(gt_labels_boyd, u_boyd_l_rw_label)
    inverse_purify_boyd_l_rw = inverse_purity_score(gt_labels_boyd, u_boyd_l_rw_label)
    NMI_boyd_l_rw = normalized_mutual_info_score(gt_labels_boyd, u_boyd_l_rw_label)

    #print('modularity for Boyd method with L_rw: ', modularity_boyd_l_rw)
    #print('ARI for Boyd method with L_rw: ', ARI_boyd_l_rw)
    #print('purify for Boyd method with L_rw: ', purify_boyd_l_rw)
    #print('inverse purify for Boyd method with L_rw: ', inverse_purify_boyd_l_rw)
    #print('NMI for Boyd method with L_rw: ', NMI_boyd_l_rw)


    sum_time_boyd_rw += time_boyd_rw
    sum_num_iteration_boyd_l_rw += num_iteration_boyd_l_rw
    sum_modularity_boyd_l_rw += modularity_boyd_l_rw
    sum_ARI_boyd_l_rw += ARI_boyd_l_rw
    sum_purify_boyd_l_rw += purify_boyd_l_rw
    sum_inverse_purify_boyd_l_rw += inverse_purify_boyd_l_rw
    sum_NMI_boyd_l_rw += NMI_boyd_l_rw
    

 
    # MMBO projection L_mix_sym
    start_time_MMBO_projection_l_sym = time.time()
    u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
    time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
    time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
    #print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

    u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
    modularity_MMBO_projection_l_sym = skn.clustering.modularity(W_MMBO ,u_MMBO_projection_l_sym_label,resolution=gamma)
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
    sum_modularity_MMBO_projection_l_sym += modularity_MMBO_projection_l_sym
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
    modularity_MMBO_projection_l_rw = skn.clustering.modularity(W_MMBO,u_MMBO_projection_l_rw_label,resolution=gamma)
    ARI_MMBO_projection_l_rw = adjusted_rand_score(u_MMBO_projection_l_rw_label, gt_labels_MMBO)
    purify_MMBO_projection_l_rw = purity_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)
    inverse_purify_MMBO_projection_l_rw = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)
    NMI_MMBO_projection_l_rw = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_rw_label)

    sum_time_MMBO_projection_rw += time_MMBO_projection_sym
    sum_num_iteration_MMBO_projection_l_rw += num_iteration_MMBO_projection_l_rw 
    sum_modularity_MMBO_projection_l_rw += modularity_MMBO_projection_l_rw
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
    modularity_mmbo_proj_B_sym = skn.clustering.modularity(W_B,u_mmbo_proj_B_sym_label,resolution=gamma)
    ARI_mmbo_proj_B_sym = adjusted_rand_score(u_mmbo_proj_B_sym_label, gt_labels_B)
    purify_mmbo_proj_B_sym = purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
    inverse_purify_mmbo_proj_B_sym = inverse_purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
    NMI_mmbo_proj_B_sym = normalized_mutual_info_score(gt_labels_B, u_mmbo_proj_B_sym_label)

    sum_time_MMBO_projection_B_sym += time_MMBO_projection_B_sym
    sum_num_repeat_mmbo_proj_B_sym += num_iteration_mmbo_proj_B_sym 
    sum_modularity_mmbo_proj_B_sym += modularity_mmbo_proj_B_sym
    sum_ARI_mmbo_proj_B_sym += ARI_mmbo_proj_B_sym
    sum_purify_mmbo_proj_B_sym += purify_mmbo_proj_B_sym
    sum_inverse_purify_mmbo_proj_B_sym += inverse_purify_mmbo_proj_B_sym
    sum_NMI_mmbo_proj_B_sym += NMI_mmbo_proj_B_sym

    # MMBO projection B_rw
    start_time_MMBO_projection_B_rw = time.time()
    u_mmbo_proj_B_rw, num_iteration_mmbo_proj_B_rw, MMBO_projection_B_rw_modularity_list = MMBO_using_projection(m, degree_W_B,  
                                            D_mmbo_B_rw, V_mmbo_B_rw, modularity_tol, u_init, W_B, gamma=gamma, stopping_condition='modularity')
    time_MMBO_projection_B_rw = time.time() - start_time_MMBO_projection_B_rw
    time_MMBO_projection_B_rw = time_eig_B_rw + time_initialize_u + time_MMBO_projection_B_rw
    #print('the number of MBO iteration for MMBO using projection with L_B_rw: ', num_repeat_mmbo_proj_B_rw)

    u_mmbo_proj_B_rw_label = vector_to_labels(u_mmbo_proj_B_rw)
    modularity_mmbo_proj_B_rw = skn.clustering.modularity(W_B,u_mmbo_proj_B_rw_label,resolution=gamma)
    ARI_mmbo_proj_B_rw = adjusted_rand_score(u_mmbo_proj_B_rw_label, gt_labels_B)
    purify_mmbo_proj_B_rw = purity_score(gt_labels_B, u_mmbo_proj_B_rw_label)
    inverse_purify_mmbo_proj_B_rw = inverse_purity_score(gt_labels_B, u_mmbo_proj_B_rw_label)
    NMI_mmbo_proj_B_rw = normalized_mutual_info_score(gt_labels_B, u_mmbo_proj_B_rw_label)

    sum_time_MMBO_projection_B_rw += time_MMBO_projection_B_rw
    sum_num_iteration_mmbo_proj_B_rw += num_iteration_mmbo_proj_B_rw 
    sum_modularity_mmbo_proj_B_rw += modularity_mmbo_proj_B_rw
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
    modularity_MMBO_using_finite_difference_sym = skn.clustering.modularity(W_MMBO,u_MMBO_using_finite_difference_sym_label,resolution=gamma)
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
    sum_modularity_MMBO_using_finite_difference_sym += modularity_MMBO_using_finite_difference_sym
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
    modularity_MMBO_using_finite_difference_rw = skn.clustering.modularity(W_MMBO,u_MMBO_using_finite_difference_rw_label,resolution=gamma)
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
    sum_modularity_MMBO_using_finite_difference_rw += modularity_MMBO_using_finite_difference_rw
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
    modularity_MMBO_using_finite_difference_B_sym = skn.clustering.modularity(W_B,u_MMBO_using_finite_difference_B_sym_label,resolution=gamma)
    ARI_MMBO_using_finite_difference_B_sym = adjusted_rand_score(u_MMBO_using_finite_difference_B_sym_label, gt_labels_B)
    purify_MMBO_using_finite_difference_B_sym = purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)
    inverse_purify_MMBO_using_finite_difference_B_sym = inverse_purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)
    NMI_MMBO_using_finite_difference_B_sym = normalized_mutual_info_score(gt_labels_B, u_MMBO_using_finite_difference_B_sym_label)

    sum_MMBO_using_finite_difference_B_sym += time_start_time_MMBO_using_finite_difference_B_sym
    sum_num_repeat_inner_nor_B_sym += num_iteration_MMBO_using_finite_difference_B_sym 
    sum_modularity_mmbo_inner_B_sym += modularity_MMBO_using_finite_difference_B_sym
    sum_ARI_mmbo_inner_B_sym += ARI_MMBO_using_finite_difference_B_sym
    sum_purify_mmbo_inner_B_sym += purify_MMBO_using_finite_difference_B_sym
    sum_inverse_purify_mmbo_inner_B_sym += inverse_purify_MMBO_using_finite_difference_B_sym
    sum_NMI_mmbo_inner_B_sym += NMI_MMBO_using_finite_difference_B_sym


    # MMBO using finite difference B_rw
    start_time_MMBO_using_finite_difference_B_rw = time.time()
    u_MMBO_using_finite_difference_B_rw, num_iertation_MMBO_using_finite_difference_B_rw, MMBO_using_finite_difference_B_rw_modularity_list = MMBO_using_finite_differendce(m,degree_W_B, 
                                        D_mmbo_B_rw, V_mmbo_B_rw, modularity_tol, N_t,  u_init, W_B, gamma=gamma, stopping_condition='modularity')
    time_MMBO_using_finite_difference_B_rw = time.time() - start_time_MMBO_using_finite_difference_B_rw
    time_MMBO_using_finite_difference_B_rw = time_eig_B_rw + time_initialize_u + time_start_time_MMBO_using_finite_difference_B_sym
    #print('the number of MBO iteration for MMBO using inner step with L_B_rw: ',num_repeat_inner_B_rw)

    u_MMBO_using_finite_difference_B_rw_label = vector_to_labels(u_MMBO_using_finite_difference_B_rw)
    modularity_MMBO_using_finite_difference_B_rw = skn.clustering.modularity(W_B,u_MMBO_using_finite_difference_B_rw_label,resolution=1)
    ARI_mmbo_inner_B_rwMMBO_using_finite_difference_B_rw = adjusted_rand_score(u_MMBO_using_finite_difference_B_rw_label, gt_labels_B)
    purify_MMBO_using_finite_difference_B_rw = purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)
    inverse_purifyMMBO_using_finite_difference_B_rw = inverse_purity_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)
    NMI_MMBO_using_finite_difference_B_rw = normalized_mutual_info_score(gt_labels_B, u_MMBO_using_finite_difference_B_rw_label)


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
average_time_MMBO_using_finite_difference_B_rw = sum_time_MMBO_using_finite_difference_B_rw / 20
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


print('Boyd method L_sym')
average_time_boyd_sym = sum_time_boyd_sym / 20
average_num_iter_boyd_sym = sum_num_iteration_boyd_l_sym / 20
average_modularity_boyd_sym = sum_modularity_boyd_l_sym / 20
average_ARI_boyd_sym = sum_ARI_boyd_l_sym / 20
average_purify_boyd_sym = sum_purify_boyd_l_sym / 20
average_inverse_purify_boyd_sym = sum_inverse_purify_boyd_l_sym / 20
average_NMI_boyd_sym = sum_NMI_boyd_l_sym / 20


print('average_time_Boyd_sym: ', average_time_boyd_sym)
print('average_num_iteration_Boyd_sym: ', average_num_iter_boyd_sym)
print('average_modularity_Boyd_sym: ', average_modularity_boyd_sym)
print('average_ARI_Boyd_sym: ', average_ARI_boyd_sym)
print('average_purify_Boyd_sym: ', average_purify_boyd_sym)
print('average_inverse_purify_Boyd_sym: ', average_inverse_purify_boyd_sym)
print('average_NMI_Boyd_sym: ', average_NMI_boyd_sym)


print('Boyd method L_rw')
average_time_boyd_rw = sum_time_boyd_rw / 20
average_num_iter_boyd_rw = sum_num_iteration_boyd_l_rw / 20
average_modularity_boyd_rw = sum_modularity_boyd_l_rw / 20
average_ARI_boyd_rw = sum_ARI_boyd_l_rw / 20
average_purify_boyd_rw = sum_purify_boyd_l_rw / 20
average_inverse_purify_boyd_rw = sum_inverse_purify_boyd_l_rw / 20
average_NMI_boyd_rw = sum_NMI_boyd_l_rw / 20


print('average_time_Boyd_rw: ', average_time_boyd_rw)
print('average_num_iteration_Boyd_rw: ', average_num_iter_boyd_rw)
print('average_modularity_Boyd_rw: ', average_modularity_boyd_rw)
print('average_ARI_Boyd_rw: ', average_ARI_boyd_rw)
print('average_purify_Boyd_rw: ', average_purify_boyd_rw)
print('average_inverse_purify_Boyd_rw: ', average_inverse_purify_boyd_rw)
print('average_NMI_Boyd_rw: ', average_NMI_boyd_rw)




# Spectral clustering with k-means
sum_time_sc=0
sum_modularity_sc =0
sum_ARI_spectral_clustering = 0
sum_purify_spectral_clustering = 0
sum_inverse_purify_spectral_clustering = 0
sum_NMI_spectral_clustering = 0

#for _ in range(20):
#    start_time_spectral_clustering = time.time()
#    sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed')
#    assignment = sc.fit_predict(W)
#    time_sc = time.time() - start_time_spectral_clustering
    #print("spectral clustering algorithm:-- %.3f seconds --" % (time_sc))

#    ass_vec = labels_to_vector(assignment)

#    modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=1)
#    ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)
#    purify_spectral_clustering = purity_score(gt_labels, assignment)
#    inverse_purify_spectral_clustering = inverse_purity_score(gt_labels, assignment)
#    NMI_spectral_clustering = normalized_mutual_info_score(gt_labels, assignment)

    #print('modularity Spectral clustering score: ', modularity_spectral_clustering)
    #print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
    #print('purify for Spectral clustering : ', purify_spectral_clustering)
    #print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
    #print('NMI for Spectral clustering: ', NMI_spectral_clustering)
    
#    sum_time_sc += time_sc
#    sum_modularity_sc += modularity_spectral_clustering
#    sum_ARI_spectral_clustering += ARI_spectral_clustering
#    sum_purify_spectral_clustering += purify_spectral_clustering
#    sum_inverse_purify_spectral_clustering += inverse_purify_spectral_clustering
#    sum_NMI_spectral_clustering += NMI_spectral_clustering

#average_time_sc = sum_time_sc / 20
#average_modularity_sc = sum_modularity_sc / 20
#average_ARI_spectral_clustering = sum_ARI_spectral_clustering / 20
#average_purify_spectral_clustering = sum_purify_spectral_clustering / 20
#average_inverse_purify_spectral_clustering = sum_inverse_purify_spectral_clustering / 20
#average_NMI_spectral_clustering = sum_NMI_spectral_clustering / 20

#print('average_time_sc: ', average_time_sc)
#print('average_modularity_sc: ', average_modularity_sc)
#print('average_ARI_spectral_clustering: ', average_ARI_spectral_clustering)
#print('average_purify_spectral_clustering: ', average_purify_spectral_clustering)
#print('average_inverse_purify_spectral_clustering: ', average_inverse_purify_spectral_clustering)
#print('average_NMI_spectral_clustering: ', average_NMI_spectral_clustering)



# CNM algorithm (can setting resolution gamma)
sum_time_CNM =0
sum_modularity_CNM =0
sum_ARI_CNM = 0
sum_purity_CNM = 0
sum_inverse_purity_CNM = 0
sum_NMI_CNM = 0

#for _ in range(20):
#    start_time_CNM = time.time()
#    partition_CNM = nx_comm.greedy_modularity_communities(G, resolution=gamma)
#    time_CNM = time.time() - start_time_CNM
    #print("CNM algorithm:-- %.3f seconds --" % (time.time() - start_time_CNM))

#    partition_CNM_list = [list(x) for x in partition_CNM]
#    partition_CNM_expand = sum(partition_CNM_list, [])

#    num_cluster_CNM = []
#    for cluster in range(len(partition_CNM_list)):
#        for number_CNM in range(len(partition_CNM_list[cluster])):
#            num_cluster_CNM.append(cluster)

#    CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))

#    partition_CNM_sort = np.sort(partition_CNM_expand)
#    CNM_list_sorted = []
#    for CNM_element in partition_CNM_sort:
#        CNM_list_sorted.append(CNM_dict[CNM_element])
#    CNM_array_sorted = np.asarray(CNM_list_sorted)
    
#    CNM_vec = labels_to_vector(CNM_array_sorted)

#    modularity_CNM = skn.clustering.modularity(W,CNM_array_sorted,resolution=gamma)
#    ARI_CNM = adjusted_rand_score(CNM_array_sorted, gt_labels)
#    purify_CNM = purity_score(gt_labels, CNM_array_sorted)
#    inverse_purify_CNM = inverse_purity_score(gt_labels, CNM_array_sorted)
#    NMI_CNM = normalized_mutual_info_score(gt_labels, CNM_array_sorted)

#    sum_time_CNM += time_CNM
#    sum_modularity_CNM += modularity_CNM
#    sum_ARI_CNM += ARI_CNM
#    sum_purity_CNM += purify_CNM
#    sum_inverse_purity_CNM += inverse_purify_CNM
#    sum_NMI_CNM += NMI_CNM


#average_time_CNM = sum_time_CNM / 20
#average_modularity_CNM = sum_modularity_CNM / 20
#average_ARI_CNM = sum_ARI_CNM / 20
#average_purity_CNM = sum_purity_CNM / 20
#average_inverse_purity_CNM = sum_inverse_purity_CNM / 20
#average_NMI_CNM = sum_NMI_CNM / 20

#print('average_time_CNM: ', average_time_CNM)
#print('average_modularity_CNM: ', average_modularity_CNM)
#print('average_ARI_CNM: ', average_ARI_CNM)
#print('average_purity_CNM: ', average_purity_CNM)
#print('average_inverse_purity_CNM: ', average_inverse_purity_CNM)
#print('average_NMI_CNM: ', average_NMI_CNM)










