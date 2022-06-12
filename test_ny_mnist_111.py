import os,sys, sklearn
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from sklearn.decomposition import PCA
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new, generate_initial_value_multiclass
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from MBO_Network import mbo_modularity_given_eig,construct_null_model
import sknetwork as skn
import networkx as nx
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1
import time
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,mbo_modularity_given_eig
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from community import community_louvain
from graph_cut.util.nystrom import nystrom_QR_l_sym, nystrom_QR_1_sym_rw, nystrom_QR_1_signed_sym_rw, nystrom_QR_signless_lap_sym
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
from sknetwork.clustering import Louvain
import graphlearning as gl



## parameter setting
dt_inner = 1
num_nodes = 69500
num_communities = 125
#m = 1 * num_communities
m = 125
dt = 1
tol = 1e-5
inner_step_count = 5
gamma = 0.02

data, gt_labels = gl.datasets.load('mnist')
pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)

W = gl.weightmatrix.knn(Z_training, 10)

# Initialize u
#start_time_initialize = time.time()
#u_init = get_initial_state_1(num_nodes, num_communities)
u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)
#time_initialize_u = time.time() - start_time_initialize
#print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


#start_time_l_sym = time.time()
#eigenvalues_hu_2, eigenvectors_hu_2, other_data, index, rw_left_eigvec, rw_right_eigvec = nystrom_QR_l_sym(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_sym = time.time() - start_time_l_sym
#print("nystrom extension in L_sym:-- %.3f seconds --" % (time_eig_l_sym))
#D_hu_2 = np.squeeze(eigenvalues_hu_2[:m])
#print('nystrom_QR D_hu: ', D_hu_2)
#V_hu = eigenvectors_hu_2[:,:m]
#V_hu_rw_left = rw_left_eigvec[:,:m]
#V_hu_rw_right = rw_right_eugvec[:,:m]
#print('nystrom_QR V_hu: ', V_hu)
#print('nystrom_QR V_hu left shape: ', V_hu_rw_left.shape)
#print('nystrom_QR V_hu left: ', V_hu_rw_left)
#print('nystrom_QR V_hu right: ', V_hu_rw_right)
#print('nystrom_QR V_hu right shape: ', V_hu_rw_right.shape)

#gt_labels_HU = gt_labels[index[500:]]
#W_HU = gl.weightmatrix.knn(other_data, 10)
#degree_W_HU = np.array(np.sum(W_HU, axis=-1)).flatten()



#start_time_l_mix = time.time()
#eigenvalues_mmbo_rw, eigenvectors_mmbo_rw, eigenvalues_mmbo_sym, eigenvectors_mmbo_sym, other_data_rw, index_rw = nystrom_QR_1_sym_rw(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#print('eigenvalues_mmbo_sym: ', eigenvalues_mmbo_sym.shape)
#D_mmbo_rw = np.squeeze(eigenvalues_mmbo_rw[:m])
#print('nystrom_QR D_mmbo_rw: ', D_mmbo_rw)
#V_mmbo_rw = eigenvectors_mmbo_rw[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)
#D_mmbo_sym = np.squeeze(eigenvalues_mmbo_sym[:m])
#print('nystrom_QR D_mmbo_sym: ', D_mmbo_sym)
#V_mmbo_sym = eigenvectors_mmbo_sym[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)


#gt_labels_rw = gt_labels[index_rw[500:]]
#W_rw = gl.weightmatrix.knn(other_data_rw, 10)
#degree_W_rw = np.array(np.sum(W_rw, axis=-1)).flatten()


#start_time_l_mix = time.time()
#eigenvalues_mmbo_B_rw, eigenvectors_mmbo_B_rw, eigenvalues_mmbo_B_sym, eigenvectors_mmbo_B_sym, other_data_B, index_B = nystrom_QR_1_signed_sym_rw(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#D_mmbo_B_rw = np.squeeze(eigenvalues_mmbo_B_rw[:m])
#print('nystrom_QR D_mmbo_rw (B^+/B^-): ', D_mmbo_B_rw)
#V_mmbo_B_rw = eigenvectors_mmbo_B_rw[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)
#D_mmbo_B_sym = np.squeeze(eigenvalues_mmbo_B_sym[:m])
#print('nystrom_QR D_mmbo_sym (B^+/B^-): ', D_mmbo_B_sym)
#V_mmbo_B_sym = eigenvectors_mmbo_B_sym[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)


#gt_labels_B = gt_labels[index_B[500:]]
#W_B = gl.weightmatrix.knn(other_data_B, 10)
#degree_W_B = np.array(np.sum(W_B, axis=-1)).flatten()



# Test HU original MBO with symmetric normalized L_F
#start_time_hu_original = time.time()
#u_hu_sym_vector, num_iter_HU_sym, HU_sym_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W_HU, dt_inner, u_init,
#                              D_hu_2, V_hu_rw_left, tol,inner_step_count, W_HU)
#time_hu_mbo = time.time() - start_time_hu_original
#print("HU original MBO:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
#print('HU original MBO the num_iteration: ', num_iter_HU)

#    u_hu_label_sym = vector_to_labels(u_hu_sym_vector)

    #HU_cluster = len(np.unique(u_hu_label_1))
    #print('the cluster Hu method found: ', HU_cluster)

#    modu_hu_original_sym = skn.clustering.modularity(W_HU,u_hu_label_sym,resolution=0.5)
#    ARI_hu_original_sym = adjusted_rand_score(u_hu_label_sym, gt_labels_HU)
#    purify_hu_original_sym = purity_score(gt_labels_HU, u_hu_label_sym)
#    inverse_purify_hu_original_sym = inverse_purity_score(gt_labels_HU, u_hu_label_sym)
#    NMI_hu_original_sym = normalized_mutual_info_score(gt_labels_HU, u_hu_label_sym)

#    print('modularity score for HU method: ', modu_hu_original_sym)
#    print('ARI for HU method: ', ARI_hu_original_sym)
#    print('purify for HU method: ', purify_hu_original_sym)
#    print('inverse purify for HU method: ', inverse_purify_hu_original_sym)
#    print('NMI for HU method: ', NMI_hu_original_sym)




## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
#    start_time_1_nor_Lf_Qh_1 = time.time()
#    u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1, MMBO_projection_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree_W_B, u_init, 
#                                            D_mmbo_B_rw, V_mmbo_B_rw, tol, W_B)
#    time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
    #print("MMBO using projection with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
    #print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_repeat_1_nor_Lf_Qh_1)

#    u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#    modularity_1_nor_lf_qh = skn.clustering.modularity(W_B,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
#    ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_labels_B)
#    purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_labels_B, u_1_nor_Lf_Qh_individual_label_1)
#    inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_labels_B, u_1_nor_Lf_Qh_individual_label_1)
#    NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_labels_B, u_1_nor_Lf_Qh_individual_label_1)

    #modularity_MMBO_projection_sym_list.append(modularity_1_nor_lf_qh)
    
#    print('modularity for MMBO using projection with L_W&P: ', modularity_1_nor_lf_qh)
#    print('ARI for MMBO using projection with L_W&P: ', ARI_mbo_1_nor_Lf_Qh_1)
#    print('purify for MMBO using projection with L_W&P: ', purify_mbo_1_nor_Lf_Qh_1)
#    print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
#    print('NMI for MMBO using projection with L_W&P: ', NMI_mbo_1_nor_Lf_Qh_1)



# MMBO1 with inner step & sym normalized L_F & Q_H
#    start_time_1_inner_nor_1 = time.time()
#u_inner_nor_1,num_repeat_inner_nor, MMBO_inner_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree_W_B, dt_inner, u_init, 
#                                        D_mmbo_B_sym, V_mmbo_B_sym, tol, inner_step_count, W_B)
#    time_MMBO_inner_step_rw = time.time() - start_time_1_inner_nor_1
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
#print('the number of MBO iteration for MMBO using inner step with L_W&P: ',num_repeat_inner_nor)

#u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
#modularity_1_inner_nor_1 = skn.clustering.modularity(W_B,u_inner_nor_label_1,resolution=0.5)
#    ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_labels_B)
#    purify_mbo_1_inner_nor_1 = purity_score(gt_labels_B, u_inner_nor_label_1)
#    inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_labels_B, u_inner_nor_label_1)
#    NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_labels_B, u_inner_nor_label_1)

    #modularity_MMBO_inner_sym_list.append(modularity_1_inner_nor_1)
    
#print('modularity for MMBO using inner step with L_W&P: ', modularity_1_inner_nor_1)
#    print('ARI for MMBO using inner step with L_W&P: ', ARI_mbo_1_inner_nor_1)
#    print('purify for MMBO using inner step with L_W&P: ', purify_mbo_1_inner_nor_1)
#    print('inverse purify for MMBO using inner step with L_W&P: ', inverse_purify_mbo_1_inner_nor_1)
#    print('NMI for MMBO using inner step with L_W&P: ', NMI_mbo_1_inner_nor_1)



# Louvain
#start_time_louvain = time.time()
#G = nx.from_scipy_sparse_array(W)
#partition_Louvain = community_louvain.best_partition(G, resolution=1)    # returns a dict
#louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#louvain_array = np.asarray(louvain_list)
#time_louvain = time.time() - start_time_louvain
#print("Louvain:-- %.3f seconds --" % (time_louvain))
#louvain_cluster = len(np.unique(louvain_array))
#print('the cluster Louvain found: ',louvain_cluster)

#modularity_louvain = skn.clustering.modularity(W,louvain_array,resolution=0.5)
#ARI_louvain = adjusted_rand_score(louvain_array, gt_labels)
#purify_louvain = purity_score(gt_labels, louvain_array)
#inverse_purify_louvain = inverse_purity_score(gt_labels, louvain_array)
#NMI_louvain = normalized_mutual_info_score(gt_labels, louvain_array)

#print(' modularity Louvain score: ', modularity_louvain)
#print(' ARI Louvain  score: ', ARI_louvain)
#print(' purify for Louvain : ', purify_louvain)
#print(' inverse purify for Louvain : ', inverse_purify_louvain)
#print(' NMI for Louvain  : ', NMI_louvain)



# CNM algorithm (can setting resolution gamma)
#G = nx.from_scipy_sparse_matrix(W)
G = nx.convert_matrix.from_scipy_sparse_matrix(W)

sum_time_CNM =0
sum_modularity_CNM =0
sum_ARI_CNM =0
sum_purify_CNM =0
sum_inverse_purify_CNM =0
sum_NMI_CNM =0

#for _ in range(5):
print('Start CNM')
start_time_CNM = time.time()
partition_CNM = nx_comm.greedy_modularity_communities(G)
time_CNM = time.time() - start_time_CNM
print("CNM algorithm:-- %.3f seconds --" % (time_CNM))

partition_CNM_list = [list(x) for x in partition_CNM]
#print(type(partition_CNM_list))

partition_CNM_expand = sum(partition_CNM_list, [])
num_cluster_CNM = []
for cluster in range(len(partition_CNM_list)):
    for number_CNM in range(len(partition_CNM_list[cluster])):
        num_cluster_CNM.append(cluster)

CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
CNM_list = list(dict.values(CNM_dict))    #convert a dict to list
CNM_array = np.asarray(CNM_list)


modularity_CNM = skn.clustering.modularity(W, CNM_array,resolution=0.5)
#    ARI_CNM = adjusted_rand_score(CNM_array, gt_labels)
#    purify_CNM = purity_score(gt_labels, CNM_array)
#    inverse_purify_CNM = inverse_purity_score(gt_labels, CNM_array)
#    NMI_CNM = normalized_mutual_info_score(gt_labels, CNM_array)

print('modularity score CNM: ', modularity_CNM)
#    print('ARI CNM: ', ARI_CNM)
#    print('purify for CNM: ', purify_CNM)
#    print('inverse purify for CNM: ', inverse_purify_CNM)
#    print('NMI for CNM: ', NMI_CNM)

#    sum_time_CNM += time_CNM
#    sum_modularity_CNM += modularity_spectral_clustering
#    sum_ARI_CNM += ARI_spectral_clustering
#    sum_purify_CNM += purify_spectral_clustering
#    sum_inverse_purify_CNM += inverse_purify_spectral_clustering
#    sum_NMI_CNM += NMI_spectral_clustering


#average_time_CNM = sum_time_CNM / 5
#average_modularity_CNM = sum_modularity_CNM / 5
#average_ARI_CNM = sum_ARI_CNM / 5
#average_purify_CNM = sum_purify_spectral_clustering / 5
#average_inverse_purify_CNM = sum_inverse_purify_CNM / 5
#average_NMI_CNM = sum_NMI_CNM / 5


#print('average_time_CNM: ', average_time_CNM)
#print('average_modularity_CNM: ', average_modularity_CNM)
#print('average_ARI_CNM: ', average_ARI_CNM)
#print('average_purify_CNM: ', average_purify_CNM)
#print('average_inverse_purify_CNM: ', average_inverse_purify_CNM)
#print('average_NMI_CNM: ', average_NMI_CNM)




# Spectral clustering with k-means

sum_time_spectral_clustering =0
sum_modularity_spectral_clustering =0
sum_ARI_spectral_clustering =0
sum_purify_spectral_clustering =0
sum_inverse_purify_spectral_clustering =0
sum_NMI_spectral_clustering =0

#for _ in range(5):
#    start_time_spectral_clustering = time.time()
#    sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed', eigen_tol=tol)
#    assignment = sc.fit_predict(W)
#    time_spectral_clustering = time.time() - start_time_spectral_clustering
#    print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

    #ass_vec = labels_to_vector(assignment)
    #ass_dict = label_to_dict (assignment)

#    modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=0.5)
#    ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)
#    purify_spectral_clustering = purity_score(gt_labels, assignment)
#    inverse_purify_spectral_clustering = inverse_purity_score(gt_labels, assignment)
#    NMI_spectral_clustering = normalized_mutual_info_score(gt_labels, assignment)

#    print('modularity Spectral clustering score(K=10 and m=K): ', modularity_spectral_clustering)
#    print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
#    print('purify for Spectral clustering : ', purify_spectral_clustering)
#    print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
#    print('NMI for Spectral clustering: ', NMI_spectral_clustering)

#    sum_time_spectral_clustering += time_spectral_clustering
#    sum_modularity_spectral_clustering += modularity_spectral_clustering
#    sum_ARI_spectral_clustering += ARI_spectral_clustering
#    sum_purify_spectral_clustering += purify_spectral_clustering
#    sum_inverse_purify_spectral_clustering += inverse_purify_spectral_clustering
#    sum_NMI_spectral_clustering += NMI_spectral_clustering


#average_time_spectral_clustering = sum_time_spectral_clustering / 5
#average_modularity_spectral_clustering  = sum_modularity_spectral_clustering / 5
#average_ARI_spectral_clustering  = sum_ARI_spectral_clustering / 5
#average_purify_spectral_clustering  = sum_purify_spectral_clustering / 5
#average_inverse_purify_spectral_clustering  = sum_inverse_purify_spectral_clustering / 5
#average_NMI_spectral_clustering = sum_NMI_spectral_clustering / 5


#print('average_time_spectral_clustering: ', average_time_spectral_clustering)
#print('average_modularity_spectral_clustering: ', average_modularity_spectral_clustering)
#print('average_ARI_spectral_clustering: ', average_ARI_spectral_clustering)
#print('average_purify_spectral_clustering: ', average_purify_spectral_clustering)
#print('average_inverse_purify_spectral_clustering: ', average_inverse_purify_spectral_clustering)
#print('average_NMI_spectral_clustering: ', average_NMI_spectral_clustering)
