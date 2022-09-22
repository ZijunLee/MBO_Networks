from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import time
import random
from community import community_louvain
from Nystrom_extension_QR import nystrom_QR_l_sym, nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
import graphlearning as gl
from MMBO_and_HU import MMBO_using_projection, MMBO_using_finite_differendce,HU_mmbo_method
from utils import vector_to_labels, labels_to_vector, purity_score, inverse_purity_score, generate_initial_value_multiclass
from Nystrom_QR_test import nystrom_QR_l_mix_sym_rw_ER_null, nystrom_QR_l_mix_B_sym_rw_ER_null


tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 0.5
tau = 0.02
num_nodes = 70000
num_communities = 120
m = num_communities


# Load MNIST data, ground truth, and build 10-nearest neighbor weight matrix
data, gt_labels = gl.datasets.load('mnist')
#gt_vec = labels_to_vector(gt_labels)


pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)





# Compute the eigenvalues and eigenvectors of L_mix_sym and L_mix_rw
eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw_ER_null(Z_training, num_nystrom=500, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]


gt_labels_MMBO = gt_labels[index_MMBO]
gt_MMBO_projection_vec = labels_to_vector(gt_labels_MMBO)
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10)
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()



u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)


u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                        E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
#time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
#time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
#print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
u_MMBO_projection_l_sym_cluster = len(np.unique(u_MMBO_projection_l_sym_label))

quantity_list = []
clusters = np.unique(u_MMBO_projection_l_sym_label)
for i in clusters:
    quantity = list(u_MMBO_projection_l_sym_label).count(i)
    quantity_list.append(quantity)
    print("The cluster %s has %s quantity" % (i, quantity))

print(np.max(quantity_list))
print(np.min(quantity_list))
