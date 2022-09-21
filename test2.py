from cgi import print_directory
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, eigsh
import networkx as nx
import random
import networkx.algorithms.community as nx_comm
import graphlearning as gl
from sklearn.decomposition import PCA
#import cv2
#import os
#import sys
#sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
#from graph_MMBO_cluster import LaplacianClustering, imageblocks, build_affinity_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh
import time
import sknetwork as skn
from sknetwork.clustering import Louvain
import cdlib
from cdlib import evaluation, NodeClustering
from cdlib.algorithms import louvain
from MMBO_and_HU import MMBO_using_projection, MMBO_using_finite_differendce, adj_to_laplacian_signless_laplacian,HU_mmbo_method, adj_to_modularity_mat
from utils import vector_to_labels, labels_to_vector, purity_score, inverse_purity_score, generate_initial_value_multiclass, label_to_dict, get_modularity_ER, dict_to_list_set
from community import community_louvain
from Nystrom_QR_test import nystrom_QR_l_mix_sym_rw_ER_null
from Nystrom_extension_QR import nystrom_QR_l_sym, nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw

tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 0.5
tau = 0.02
num_nodes = 70000
num_communities = 120
#num_communities = 10
m = num_communities

data, gt_labels = gl.datasets.load('mnist')
#gt_vec = labels_to_vector(gt_labels)

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


eigenvalues_sym, eigenvectors_sym, rw_left_eigvec, order_raw_data_HU, index_HU, time_eig_l_sym, time_eig_l_rw = nystrom_QR_l_sym(Z_training, num_nystrom=500, tau=tau)
eig_val_HU_sym = np.squeeze(eigenvalues_sym[:m])
eig_vec_HU_sym = eigenvectors_sym[:,:m]
eig_vec_HU_rw = rw_left_eigvec[:,:m]

gt_labels_HU = gt_labels[index_HU]
W_HU = gl.weightmatrix.knn(order_raw_data_HU, 10)
degree_W_HU = np.array(np.sum(W_HU, axis=-1)).flatten()
gt_HU_vec = labels_to_vector(gt_labels_HU)
print('eig_val_HU_sym', eig_val_HU_sym)



eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw(Z_training, num_nystrom=500, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]

gt_labels_MMBO = gt_labels[index_MMBO]
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10)
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()
gt_MMBO_projection_vec = labels_to_vector(gt_labels_MMBO)

u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)
print('u_init', u_init.shape)


#expand_zero_columns = np.zeros((num_nodes, num_communities - 10))
#print('expand_zero_columns', expand_zero_columns.shape)

#gt_vec_new = np.append(gt_vec, expand_zero_columns, axis=1)
#print('gt_vec_new', gt_vec_new.shape)

#u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=10)
#print('u_init', u_init.shape)



u_hu_sym_vector, num_iteration_HU_sym, HU_sym_modularity_list = HU_mmbo_method(num_nodes, degree_W_HU, eig_val_HU_sym, eig_vec_HU_sym,
                                 modularity_tol, N_t, u_init, W_HU, gamma=gamma, stopping_condition='modularity') 

#print('the num_iteration of HU method with L_sym: ', num_iteration_HU_sym)

u_hu_sym_label = vector_to_labels(u_hu_sym_vector)
u_hu_sym_label_cluster = len(np.unique(u_hu_sym_label))
print('u_hu_sym_label_cluster', u_hu_sym_label_cluster)

modularity_hu_sym = skn.clustering.modularity(W_HU, u_hu_sym_label,resolution=gamma)
ARI_hu_sym = adjusted_rand_score(u_hu_sym_label, gt_labels_HU)
purify_hu_sym = purity_score(gt_labels_HU, u_hu_sym_label)
inverse_purify_hu_sym = inverse_purity_score(gt_labels_HU, u_hu_sym_label)
NMI_hu_sym = normalized_mutual_info_score(gt_labels_HU, u_hu_sym_label)

print('modularity score for HU method: ', modularity_hu_sym)
print('ARI for HU method: ', ARI_hu_sym)
print('purify for HU method: ', purify_hu_sym)
print('inverse purify for HU method: ', inverse_purify_hu_sym)
print('NMI for HU method: ', NMI_hu_sym)



u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
#time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
#time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
#print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
u_MMBO_projection_l_sym_dict = label_to_dict(u_MMBO_projection_l_sym_label)
u_MMBO_projection_l_sym_list = dict_to_list_set(u_MMBO_projection_l_sym_dict)
u_MMBO_projection_l_sym_coms = NodeClustering(u_MMBO_projection_l_sym_list, graph=None)
u_MMBO_projection_cluster = len(np.unique(u_MMBO_projection_l_sym_label))
print('u_MMBO_projection_cluster', u_MMBO_projection_cluster)

#ER_modularity_MMBO_projection_l_sym = evaluation.erdos_renyi_modularity(G,u_MMBO_projection_l_sym_coms)[2]
#modularity_MMBO_projection_l_sym = evaluation.newman_girvan_modularity(G,u_MMBO_projection_l_sym_coms)[2]
#ER_modularity_MMBO_projection_l_sym_1 = get_modularity_ER(W_MMBO, u_MMBO_projection_l_sym_dict, gamma=gamma)

modularity_MMBO_projection_l_sym = skn.clustering.modularity(W_MMBO ,u_MMBO_projection_l_sym_label,resolution=gamma)
ARI_MMBO_projection_l_sym = adjusted_rand_score(u_MMBO_projection_l_sym_label, gt_labels_MMBO)
purify_MMBO_projection_l_sym = purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
inverse_purify_MMBO_projection_l_sym = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
NMI_MMBO_projection_l_sym = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)


#print('ER_modularity_MMBO_projection_l_sym_1', ER_modularity_MMBO_projection_l_sym_1)
#print('ER modularity for MMBO using projection with L_W&P: ', ER_modularity_MMBO_projection_l_sym)
print('modularity for MMBO using projection with L_W&P: ', modularity_MMBO_projection_l_sym)
print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)




u_init_sup = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=10)
row_numbers = range(0, len(gt_labels))
Rs = random.sample(row_numbers, 7000)

u_init_HU_sup = u_init_sup
u_init_HU_sup[[Rs],:] = gt_HU_vec[[Rs],:]

u_init_LWP_sup = u_init_sup
u_init_LWP_sup[[Rs],:] = gt_MMBO_projection_vec[[Rs],:]

#print('u_init_sup', u_init_sup.shape)



u_hu_sym_vector, num_iteration_HU_sym, HU_sym_modularity_list = HU_mmbo_method(num_nodes, degree_W_HU, eig_val_HU_sym, eig_vec_HU_sym,
                                 modularity_tol, N_t, u_init_HU_sup, W_HU, gamma=gamma, stopping_condition='modularity') 

#print('the num_iteration of HU method with L_sym: ', num_iteration_HU_sym)

u_hu_sym_label = vector_to_labels(u_hu_sym_vector)
u_hu_sym_label_cluster = len(np.unique(u_hu_sym_label))
print('u_hu_sym_label_cluster', u_hu_sym_label_cluster)

modularity_hu_sym = skn.clustering.modularity(W_HU, u_hu_sym_label,resolution=gamma)
ARI_hu_sym = adjusted_rand_score(u_hu_sym_label, gt_labels_HU)
purify_hu_sym = purity_score(gt_labels_HU, u_hu_sym_label)
inverse_purify_hu_sym = inverse_purity_score(gt_labels_HU, u_hu_sym_label)
NMI_hu_sym = normalized_mutual_info_score(gt_labels_HU, u_hu_sym_label)


print('NG -- 10%, K=10')
print('modularity score for HU method: ', modularity_hu_sym)
print('ARI for HU method: ', ARI_hu_sym)
print('purify for HU method: ', purify_hu_sym)
print('inverse purify for HU method: ', inverse_purify_hu_sym)
print('NMI for HU method: ', NMI_hu_sym)



u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init_LWP_sup, W_MMBO, gamma=gamma, stopping_condition='modularity') 
#time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
#time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
#print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
u_MMBO_projection_l_sym_dict = label_to_dict(u_MMBO_projection_l_sym_label)
u_MMBO_projection_l_sym_list = dict_to_list_set(u_MMBO_projection_l_sym_dict)
#u_MMBO_projection_l_sym_coms = NodeClustering(u_MMBO_projection_l_sym_list, graph=None)
u_MMBO_projection_cluster = len(np.unique(u_MMBO_projection_l_sym_label))
print('u_MMBO_projection_cluster', u_MMBO_projection_cluster)

#ER_modularity_MMBO_projection_l_sym = evaluation.erdos_renyi_modularity(G,u_MMBO_projection_l_sym_coms)[2]
#modularity_MMBO_projection_l_sym = evaluation.newman_girvan_modularity(G,u_MMBO_projection_l_sym_coms)[2]
#ER_modularity_MMBO_projection_l_sym_1 = get_modularity_ER(W_MMBO, u_MMBO_projection_l_sym_dict, gamma=gamma)

modularity_MMBO_projection_l_sym = skn.clustering.modularity(W_MMBO ,u_MMBO_projection_l_sym_label,resolution=gamma)
ARI_MMBO_projection_l_sym = adjusted_rand_score(u_MMBO_projection_l_sym_label, gt_labels_MMBO)
purify_MMBO_projection_l_sym = purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
inverse_purify_MMBO_projection_l_sym = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
NMI_MMBO_projection_l_sym = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)

print('NG -- 10%, K=10')
#print('ER_modularity_MMBO_projection_l_sym_1', ER_modularity_MMBO_projection_l_sym_1)
#print('ER modularity for MMBO using projection with L_W&P: ', ER_modularity_MMBO_projection_l_sym)
print('modularity for MMBO using projection with L_W&P: ', modularity_MMBO_projection_l_sym)
print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)