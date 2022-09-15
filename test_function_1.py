from cdlib.algorithms import louvain
from cdlib import evaluation
import networkx as nx
from community import community_louvain
import numpy as np
from MMBO_and_HU import HU_mmbo_method, adj_to_laplacian_signless_laplacian, MMBO_using_projection
from utils import generate_initial_value_multiclass, label_to_dict,dict_to_list_set, purity_score, inverse_purity_score
from scipy.sparse.linalg import eigsh
import networkx.algorithms.community as nx_comm
from cdlib import NodeClustering
import cdlib
import sknetwork as skn
from sknetwork.data import karate_club
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import graphlearning as gl
from sklearn.decomposition import PCA
import time
from Nystrom_QR_test import nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw
from utils import vector_to_labels, dict_to_list_set, label_to_dict


## parameter setting
num_nodes = 70000
num_communities = 120
m = 1 * num_communities
#m = 100
dt = 1
tol = 1e-5
modularity_tol = 1e-4
inner_step_count = 5
gamma = 0.5
tau = 0.02
num_nystrom = 500

#Load labels, knndata, and build 10-nearest neighbor weight matrix
#W = gl.weightmatrix.knn('mnist', 10)


data, gt_labels = gl.datasets.load('mnist')

pca = PCA(n_components = 50,svd_solver='arpack')
Z_training = pca.fit_transform(data)


W = gl.weightmatrix.knn(Z_training, 10)
#degree_W = np.array(np.sum(W, axis=-1)).flatten()
#print('adj_mat type: ', type(adj_mat))
G = nx.convert_matrix.from_scipy_sparse_matrix(W)

# Construct the Erdos-Renyi null model P
total_degree = np.sum(W, dtype=np.int64)
probability = total_degree / (num_nodes * (num_nodes -1))

all_one_matrix = np.ones((num_nodes, num_nodes))
#diag_matrix = np.diag(np.full(num_nodes,1))
ER_null_adj = probability * all_one_matrix
#print('ER_null_adj')

del all_one_matrix

start_time_create_ER_graph = time.time()
#G_ER = nx.fast_gnp_random_graph(num_nodes, probability)
#ER_null_adj = nx.convert_matrix.to_numpy_array(G_ER)
print("creat Erdos-Renyi graph:-- %.3f seconds --" % (time.time() - start_time_create_ER_graph))

#del G_ER

ER_null_first_k_columns = ER_null_adj[:, :num_nystrom]

del ER_null_adj

# Initialize u
start_time_initialize = time.time()
u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))



eig_val_MMBO_sym, eig_vec_MMBO_sym, eig_val_MMBO_rw, eig_vec_MMBO_rw, order_raw_data_MMBO, index_MMBO, time_eig_l_mix_sym, time_eig_l_mix_rw = nystrom_QR_l_mix_sym_rw(Z_training, ER_null_first_k_columns, num_nystrom=num_nystrom, tau = tau)
E_mmbo_sym = np.squeeze(eig_val_MMBO_sym[:m])
V_mmbo_sym = eig_vec_MMBO_sym[:,:m]
E_mmbo_rw = np.squeeze(eig_val_MMBO_rw[:m])
V_mmbo_rw = eig_vec_MMBO_rw[:,:m]

print('E_mmbo_sym: ', E_mmbo_sym)
print('E_mmbo_rw: ', E_mmbo_rw)

gt_labels_MMBO = gt_labels[index_MMBO]
W_MMBO = gl.weightmatrix.knn(order_raw_data_MMBO, 10)
degree_W_MMBO = np.array(np.sum(W_MMBO, axis=-1)).flatten()



eig_val_mmbo_B_sym, eig_vec_mmbo_B_sym, eig_val_mmbo_B_rw, eig_vec_mmbo_B_rw, order_raw_data_B, index_B, time_eig_B_sym, time_eig_B_rw = nystrom_QR_l_mix_B_sym_rw(Z_training, ER_null_first_k_columns, num_nystrom=num_nystrom, tau=tau)
D_mmbo_B_sym = np.squeeze(eig_val_mmbo_B_sym[:m])
V_mmbo_B_sym = eig_vec_mmbo_B_sym[:,:m]
D_mmbo_B_rw = np.squeeze(eig_val_mmbo_B_rw[:m])
V_mmbo_B_rw = eig_vec_mmbo_B_rw[:,:m]

print('D_mmbo_B_sym: ', D_mmbo_B_sym)
print('D_mmbo_B_rw: ', D_mmbo_B_rw)

gt_labels_B = gt_labels[index_B]
W_B = gl.weightmatrix.knn(order_raw_data_B, 10)
degree_W_B = np.array(np.sum(W_B, axis=-1)).flatten()



start_time_MMBO_projection_l_sym = time.time()
u_MMBO_projection_l_sym, num_iteration_MMBO_projection_l_sym, MMBO_projection_sym_modularity_list = MMBO_using_projection(m, degree_W_MMBO,  
                                        E_mmbo_sym, V_mmbo_sym, modularity_tol, u_init, W_MMBO, gamma=gamma, stopping_condition='modularity') 
time_MMBO_projection_sym = time.time() - start_time_MMBO_projection_l_sym
time_MMBO_projection_sym = time_eig_l_mix_sym + time_initialize_u + time_MMBO_projection_sym
print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_iteration_MMBO_projection_l_sym)

u_MMBO_projection_l_sym_label = vector_to_labels(u_MMBO_projection_l_sym)
u_MMBO_projection_l_sym_dict = label_to_dict(u_MMBO_projection_l_sym_label)
u_MMBO_projection_l_sym_list = dict_to_list_set(u_MMBO_projection_l_sym_dict)
u_MMBO_projection_l_sym_coms = NodeClustering(u_MMBO_projection_l_sym_list, graph=None)

ER_modularity_MMBO_projection_l_sym = evaluation.erdos_renyi_modularity(G,u_MMBO_projection_l_sym_coms)[2]
#modularity_MMBO_projection_l_sym = evaluation.newman_girvan_modularity(G,u_MMBO_projection_l_sym_coms)[2]
modularity_MMBO_projection_l_sym = skn.clustering.modularity(W_MMBO ,u_MMBO_projection_l_sym_label,resolution=gamma)
ARI_MMBO_projection_l_sym = adjusted_rand_score(u_MMBO_projection_l_sym_label, gt_labels_MMBO)
purify_MMBO_projection_l_sym = purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
inverse_purify_MMBO_projection_l_sym = inverse_purity_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)
NMI_MMBO_projection_l_sym = normalized_mutual_info_score(gt_labels_MMBO, u_MMBO_projection_l_sym_label)

print('ER-modularity for MMBO using projection with L_W&P: ', ER_modularity_MMBO_projection_l_sym)
print('modularity for MMBO using projection with L_W&P: ', modularity_MMBO_projection_l_sym)
print('ARI for MMBO using projection with L_W&P: ', ARI_MMBO_projection_l_sym)
print('purify for MMBO using projection with L_W&P: ', purify_MMBO_projection_l_sym)
print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_MMBO_projection_l_sym)
print('NMI for MMBO using projection with L_W&P: ', NMI_MMBO_projection_l_sym)


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
modularity_mmbo_proj_B_sym = skn.clustering.modularity(W_B,u_mmbo_proj_B_sym_label,resolution=gamma)
ARI_mmbo_proj_B_sym = adjusted_rand_score(u_mmbo_proj_B_sym_label, gt_labels_B)
purify_mmbo_proj_B_sym = purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
inverse_purify_mmbo_proj_B_sym = inverse_purity_score(gt_labels_B, u_mmbo_proj_B_sym_label)
NMI_mmbo_proj_B_sym = normalized_mutual_info_score(gt_labels_B, u_mmbo_proj_B_sym_label)

print('ER-modularity for MMBO using projection with B_sym: ', ER_modularity_mmbo_proj_B_sym)
print('modularity for MMBO using projection with B_sym: ', modularity_mmbo_proj_B_sym)
print('ARI for MMBO using projection with B_sym: ', ARI_mmbo_proj_B_sym)
print('purify for MMBO using projection with B_sym: ', purify_mmbo_proj_B_sym)
print('inverse purify for MMBO using projection with B_sym: ', inverse_purify_mmbo_proj_B_sym)
print('NMI for MMBO using projection with B_sym: ', NMI_mmbo_proj_B_sym)