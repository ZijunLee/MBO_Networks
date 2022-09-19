import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, eigsh
import networkx as nx
import random
import networkx.algorithms.community as nx_comm
import graphlearning as gl
from sklearn.decomposition import PCA
import cv2
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from graph_MMBO_cluster import LaplacianClustering, imageblocks, build_affinity_matrix
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

tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 1
tau = 0.02
num_nodes = 70000
num_communities = 120
m = num_communities

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


eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw_ER_null(Z_training, num_nystrom=500, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]

gt_labels_MMBO = gt_labels[index_MMBO]
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10)
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()

u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)

u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                            E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
#time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
#time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
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

print('ER modularity for MMBO using projection with L_W&P: ', ER_modularity_MMBO_projection_l_sym)
print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)