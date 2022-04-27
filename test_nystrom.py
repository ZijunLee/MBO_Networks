import os,sys, sklearn
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import scipy as sp
import time
import graphlearning as gl
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import sknetwork as skn
import networkx as nx
from community import community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1, labels_to_vector
from graph_cut.util.nystrom import nystrom_extension, nystrom_extension_test, nystrom_new
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,construct_null_model
from graph_mbo.utils import vector_to_labels, purity_score,inverse_purity_score



## parameter setting
dt_inner = 1
num_nodes = 70000
num_communities = 10
m = 5 * num_communities
#m = 100
dt = 1
tol = 1e-5
inner_step_count = 5
gamma = 0.02

#Load labels, knndata, and build 10-nearest neighbor weight matrix
#W = gl.weightmatrix.knn('mnist', 10)


data, gt_labels = gl.datasets.load('mnist')

pca = PCA(n_components = 50,svd_solver='full')
Z_training = pca.fit_transform(data)

#n1, p = Z_training.shape
#print("Features:", p)

W = gl.weightmatrix.knn(Z_training, 10)
degree_W = np.array(np.sum(W, axis=-1)).flatten()
#print('adj_mat type: ', type(adj_mat))

# Initialize u
start_time_initialize = time.time()
u_init = get_initial_state_1(num_nodes, num_communities)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


start_time_l_sym = time.time()
eigenvalues_1, eigenvectors_1 = nystrom_new(Z_training, num_nystrom=500, gamma=gamma)
D_hu = np.squeeze(eigenvalues_1[:m])
V_hu = eigenvectors_1[:,:m]
time_eig_l_sym = time.time() - start_time_l_sym
print("nystrom extension in L_sym:-- %.3f seconds --" % (time_eig_l_sym))


start_time_l_mix = time.time()
eigenvalues_1, eigenvectors_1 = nystrom_extension(Z_training, num_nystrom=500, gamma=gamma)
D_mmbo = np.squeeze(eigenvalues_1[1:m+1]) 
V_mmbo = eigenvectors_1[:,1:m+1]
#D_hu = np.squeeze(eigenvalues_1[:m]) 
#V_hu = eigenvectors_1[:,:m]
time_eig_l_mix = time.time() - start_time_l_mix
print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))


# Test HU original MBO with symmetric normalized L_F
start_time_hu_original = time.time()
u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W, dt_inner, u_init,
                             D_hu, V_hu, tol,inner_step_count) 
time_hu_mbo = time.time() - start_time_hu_original
print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
print('the num_iteration of HU method: ', num_iter_HU)

u_hu_label_1 = vector_to_labels(u_hu_vector)

modu_hu_original_1 = skn.clustering.modularity(W,u_hu_label_1,resolution=0.5)
ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_labels)
purify_hu_original_1 = purity_score(gt_labels, u_hu_label_1)
inverse_purify_hu_original_1 = inverse_purity_score(gt_labels, u_hu_label_1)
NMI_hu_original_1 = normalized_mutual_info_score(gt_labels, u_hu_label_1)

print('modularity score for HU method: ', modu_hu_original_1)
print('ARI for HU method: ', ARI_hu_original_1)
print('purify for HU method: ', purify_hu_original_1)
print('inverse purify for HU method: ', inverse_purify_hu_original_1)
print('NMI for HU method: ', NMI_hu_original_1)



## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
start_time_1_nor_Lf_Qh_1 = time.time()
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m, degree_W, u_init, 
                                                 D_mmbo, V_mmbo, tol)
time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1                                                
print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)

u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

modularity_1_nor_lf_qh = skn.clustering.modularity(W,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)

print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)
print('ARI for MMBO using projection with L_{mix}: ', ARI_mbo_1_nor_Lf_Qh_1)
print('purify for MMBO using projection with L_{mix}: ', purify_mbo_1_nor_Lf_Qh_1)
print('inverse purify for MMBO using projection with L_{mix}: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print('NMI for MMBO using projection with L_{mix}: ', NMI_mbo_1_nor_Lf_Qh_1)



# MMBO1 with inner step & sym normalized L_F & Q_H
start_time_1_inner_nor_1 = time.time()
u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes, num_communities, m, dt_inner, u_init, 
                                        D_mmbo, V_mmbo, tol, inner_step_count)
time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
print("MMBO using inner step with L_{mix}:-- %.3f seconds --" % ( time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
print('the number of MBO iteration for MMBO using inner step with L_{mix}: ',num_repeat_inner_nor)

u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)

modularity_1_inner_nor_1 = skn.clustering.modularity(W,u_inner_nor_label_1,resolution=0.5)
ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_labels)
purify_mbo_1_inner_nor_1 = purity_score(gt_labels, u_inner_nor_label_1)
inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_labels, u_inner_nor_label_1)
NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_labels, u_inner_nor_label_1)

print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_1)
print('ARI for MMBO using inner step with L_{mix}: ', ARI_mbo_1_inner_nor_1)
print('purify for MMBO using inner step with L_{mix}: ', purify_mbo_1_inner_nor_1)
print('inverse purify for MMBO using inner step with L_{mix}: ', inverse_purify_mbo_1_inner_nor_1)
print('NMI for MMBO using inner step with L_{mix}: ', NMI_mbo_1_inner_nor_1)


# Louvain
start_time_louvain = time.time()
#G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#G = nx.convert_matrix.from_numpy_array(adj_mat)
#partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
#louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#louvain_array = np.asarray(louvain_list)
#print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))
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


# Spectral clustering with k-means
#start_time_spectral_clustering = time.time()
#sc = SpectralClustering(n_clusters=num_communities, gamma=gamma, affinity='precomputed')
#assignment = sc.fit_predict(W)
#print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

#ass_vec = labels_to_vector(assignment)
#ass_dict = label_to_dict (assignment)

#modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=0.5)
#ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)
#purify_spectral_clustering = purity_score(gt_labels, assignment)
#inverse_purify_spectral_clustering = inverse_purity_score(gt_labels, assignment)
#NMI_spectral_clustering = normalized_mutual_info_score(gt_labels, assignment)

#print(' modularity Spectral clustering score(K=10 and m=K): ', modularity_spectral_clustering)
#print(' ARI Spectral clustering  score: ', ARI_spectral_clustering)
#print(' purify for Spectral clustering : ', purify_spectral_clustering)
#print(' inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
#print(' NMI for Spectral clustering: ', NMI_spectral_clustering)
