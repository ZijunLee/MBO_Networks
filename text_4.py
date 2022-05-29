import os,sys, sklearn
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import scipy as sp
from graph_cut.data.read_mnist import Read_mnist_function, subsample
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from sklearn.decomposition import PCA
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new, generate_initial_value_multiclass
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score,f1_score
from MBO_Network import mbo_modularity_given_eig,construct_null_model
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1
import time
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,mbo_modularity_given_eig
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from community import community_louvain
from graph_cut.util.nystrom import nystrom_extension_test, nystrom_new, nystrom_QR, nystrom_QR_l_sym, nystrom_QR_B_signed, nystrom_QR_1, nystrom_QR_1_signed, nystrom_QR_1_random_walk, nystrom_QR_1_sym_rw, nystrom_QR_1_signed_sym_rw
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
m = 180
dt = 1
tol = 1e-5
inner_step_count = 5
gamma = 0.02

#gpath = '/'.join(os.getcwd().split('/')[:-1])

#raw_data, labels = Read_mnist(digits = [4,9],path = gpath+'/MBO_signed_graphs/graph_cut/data') 
#raw_data = raw_data/255.
#full_data, full_labels = Read_mnist_function(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
#full_data, full_labels = Read_mnist(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
#full_data = full_data/255.
#sample_data,sample_labels = subsample(sample_num = 300, rd = full_data, labels = full_labels)
#print('sample number is 3000')
#print('sample_labels: ',sample_labels)


data, gt_labels = gl.datasets.load('mnist')
#print('raw data shape: ', data.shape)


#pca = PCA(n_components = 50,svd_solver='full')
#pca.fit(full_data)
#train_data = pca.transform(sample_data)

pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)
#print('Z_training shape: ', Z_training.shape)


#W = gl.weightmatrix.knn(Z_training, 10)
#print('W type: ', type(W))
#degree_W = np.array(np.sum(W, axis=-1)).flatten()
#adj_mat = W.toarray()



#adj_mat = build_affinity_matrix_new(Z_training,affinity='rbf',gamma=gamma, n_neighbors=10, neighbor_type='knearest')
#print('dist_matrix type: ',type(dist_matrix))

#degree = np.array(np.sum(adj_mat, axis=-1)).flatten()
#null_model = construct_null_model(adj_mat)

#num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, num_communities,m ,target_size=None)
#l_mix =  sym_graph_lap + sym_signless_lap


# Initialize u
start_time_initialize = time.time()
#u_init = get_initial_state_1(num_nodes, num_communities)
u_init = generate_initial_value_multiclass('rd_equal', n_samples=num_nodes, n_class=num_communities)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


#start_time_eigendecomposition = time.time()
#eigenvalues_mmbo, eigenvectors_mmbo = nystrom_extension(Z_training, num_nystrom=500, gamma=gamma)
#D_mmbo = np.squeeze(eigenvalues_mmbo[1:m+1]) 
#V_mmbo = eigenvectors_mmbo[:,1:m+1]
#D_hu = np.squeeze(eigenvalues_1[:m]) 
#V_hu = eigenvectors_1[:,:m]
#time_eig_l_mix = time.time() - start_time_eigendecomposition
#print("compute eigendecomposition:-- %.3f seconds --" % (time_eig_l_mix))
#print('nystrom D_sign_1 shape: ', D_hu)
#print('nystrom V_sign_1 shape: ', V_hu)


#start_time_l_sym = time.time()
#eigenvalues_hu_1, eigenvectors_hu_1 = nystrom_new(Z_training, num_nystrom=500, gamma=gamma)
#D_hu_2 = np.squeeze(eigenvalues_hu_1[:m])
#V_hu_1 = eigenvectors_hu_1[:,:m]
#time_eig_l_sym = time.time() - start_time_l_sym
#print("nystrom extension in L_sym:-- %.3f seconds --" % (time_eig_l_sym))
#print('nystrom D_sign_1 shape: ', D_hu_1)
#print('nystrom V_sign_1 shape: ', V_hu_1)


#start_time_l_mix = time.time()
#eigenvalues_mmbo, eigenvectors_mmbo = nystrom_extension_test(Z_training, num_nystrom=500, gamma=gamma)
#D_mmbo_sym = np.squeeze(eigenvalues_mmbo[:m])
#V_mmbo_sym = eigenvectors_mmbo[:,:m]
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#print('nystrom D_sign_1 shape: ', D_mmbo)
#print('nystrom V_sign_1 shape: ', V_hu)


#start_time_l_sym = time.time()
eigenvalues_hu_2, eigenvectors_hu_2, other_data, index, rw_left_eigvec, rw_right_eigvec = nystrom_QR_l_sym(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_sym = time.time() - start_time_l_sym
#print("nystrom extension in L_sym:-- %.3f seconds --" % (time_eig_l_sym))
D_hu_2 = np.squeeze(eigenvalues_hu_2[:m])
#D_hu_2 = np.where(D_hu_2 > 0, D_hu_2, 0)
#print('nystrom_QR D_hu: ', D_hu_2)
V_hu = eigenvectors_hu_2[:,:m]
V_hu_rw_left = rw_left_eigvec[:,:m]
#V_hu_rw_right = rw_right_eugvec[:,:m]
#V_multi_1 = V_hu_rw_left * V_hu_rw_right
#V_multi_2 = V_hu_rw_right * V_hu_rw_left
#print('nystrom_QR V_hu: ', V_hu)
#print('nystrom_QR V_hu left shape: ', V_hu_rw_left.shape)
#print('nystrom_QR V_hu left: ', V_hu_rw_left)
#print('nystrom_QR V_hu right: ', V_hu_rw_right)
#print('nystrom_QR V_hu multi_1: ', V_multi_1)
#print('nystrom_QR V_hu multi_2: ', V_multi_2)
#print('nystrom_QR V_hu right shape: ', V_hu_rw_right.shape)

gt_labels_HU = gt_labels[index[500:]]
W_HU = gl.weightmatrix.knn(other_data, 10)
degree_W_HU = np.array(np.sum(W_HU, axis=-1)).flatten()


#start_time_l_mix = time.time()
#eigenvalues_mmbo_1, eigenvectors_mmbo_1, other_data, index = nystrom_QR_1(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#D_mmbo_1 = np.squeeze(eigenvalues_mmbo_1[:m])
#print('nystrom_QR D_mmbo_1: ', D_mmbo_1)
#V_mmbo_1 = eigenvectors_mmbo_1[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)


#start_time_l_mix = time.time()
#eigenvalues_mmbo_2, eigenvectors_mmbo_2, other_data, index = nystrom_QR_1_signed(Z_training, num_nystrom=500, gamma=gamma)
#D_mmbo_2 = np.squeeze(eigenvalues_mmbo_2[:m])
#print('nystrom_QR D_mmbo_2: ', D_mmbo_2)
#V_mmbo_2 = eigenvectors_mmbo_2[:,:m]
#print('nystrom_QR V_mmbo_2: ', V_mmbo_2)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))

#gt_labels_sym = gt_labels[index[500:]]
#W = gl.weightmatrix.knn(other_data, 10)
#degree_W = np.array(np.sum(W, axis=-1)).flatten()


#start_time_l_mix = time.time()
eigenvalues_mmbo_rw, eigenvectors_mmbo_rw, eigenvalues_mmbo_sym, eigenvectors_mmbo_sym, other_data_rw, index_rw = nystrom_QR_1_sym_rw(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#print('eigenvalues_mmbo_sym: ', eigenvalues_mmbo_sym.shape)
D_mmbo_rw = np.squeeze(eigenvalues_mmbo_rw[:m])
#D_mmbo_rw = np.insert(D_mmbo_rw,0,0)
#print('nystrom_QR D_mmbo_rw: ', D_mmbo_rw)
V_mmbo_rw = eigenvectors_mmbo_rw[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)
D_mmbo_sym = np.squeeze(eigenvalues_mmbo_sym[:m])
#D_mmbo_sym = np.insert(D_mmbo_sym,0,0)
#print('nystrom_QR D_mmbo_sym: ', D_mmbo_sym)
V_mmbo_sym = eigenvectors_mmbo_sym[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)


gt_labels_rw = gt_labels[index_rw[500:]]
W_rw = gl.weightmatrix.knn(other_data_rw, 10)
degree_W_rw = np.array(np.sum(W_rw, axis=-1)).flatten()


#start_time_l_mix = time.time()
eigenvalues_mmbo_B_rw, eigenvectors_mmbo_B_rw, eigenvalues_mmbo_B_sym, eigenvectors_mmbo_B_sym, other_data_B, index_B = nystrom_QR_1_signed_sym_rw(Z_training, num_nystrom=500, gamma=gamma)
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
D_mmbo_B_rw = np.squeeze(eigenvalues_mmbo_B_rw[:m])
#D_mmbo_B_rw = np.where(D_mmbo_B_rw > 0, D_mmbo_B_rw, 0)
#D_mmbo_B_rw = np.insert(D_mmbo_B_rw,0,0)
#print('nystrom_QR D_mmbo_rw (B^+/B^-): ', D_mmbo_B_rw)
#D_mmbo_B_rw = np.insert(D_mmbo_B_rw,0,0)
V_mmbo_B_rw = eigenvectors_mmbo_rw[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)
D_mmbo_B_sym = np.squeeze(eigenvalues_mmbo_B_sym[:m])
#D_mmbo_B_sym = np.where(D_mmbo_B_sym > 0, D_mmbo_B_sym, 0)
#D_mmbo_B_sym = np.insert(D_mmbo_B_sym,0,0)
#print('nystrom_QR D_mmbo_sym (B^+/B^-): ', D_mmbo_B_sym)
V_mmbo_B_sym = eigenvectors_mmbo_sym[:,:m]
#print('nystrom_QR V_mmbo_1: ', V_mmbo_1)


gt_labels_B = gt_labels[index_B[500:]]
W_B = gl.weightmatrix.knn(other_data_B, 10)
degree_W_B = np.array(np.sum(W_B, axis=-1)).flatten()


# Louvain
#start_time_louvain = time.time()
#G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#G = nx.convert_matrix.from_numpy_array(W)
#partition_Louvain = community_louvain.best_partition(G, resolution=1)    # returns a dict
#louvain = Louvain(modularity='newman')
#louvain_array = louvain.fit_transform(W)
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

#m = 1 * louvain_cluster



# Test HU original MBO with symmetric normalized L_F
u_hu_vector, num_iter_HU, HU_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W_HU, dt_inner, u_init,
                             D_hu_2, V_hu, tol,inner_step_count, W_HU)

u_hu_label_1 = vector_to_labels(u_hu_vector)

#HU_cluster = len(np.unique(u_hu_label_1))
#print('the cluster Hu method found: ', HU_cluster)

modu_hu_original_1 = skn.clustering.modularity(W_HU,u_hu_label_1,resolution=0.5)
#ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_labels)
#purify_hu_original_1 = purity_score(gt_labels, u_hu_label_1)
#inverse_purify_hu_original_1 = inverse_purity_score(gt_labels, u_hu_label_1)
#NMI_hu_original_1 = normalized_mutual_info_score(gt_labels, u_hu_label_1)

print('modularity score for HU method: ', modu_hu_original_1)
#print('ARI for HU method: ', ARI_hu_original_1)
#print('purify for HU method: ', purify_hu_original_1)
#print('inverse purify for HU method: ', inverse_purify_hu_original_1)
#print('NMI for HU method: ', NMI_hu_original_1)


#start_time_hu_original = time.time()
#u_hu_vector_left, num_iter_HU, HU_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W_HU, dt_inner, u_init,
#                             D_hu_2, V_hu_rw_left, tol,inner_step_count, W_HU)

#u_hu_vector_right, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W, dt_inner, u_init,
#                            D_hu_2, V_hu_rw_right, tol,inner_step_count) 
#time_hu_mbo = time.time() - start_time_hu_original
#print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
#print('the num_iteration of HU method: ', num_iter_HU)

#u_hu_label_left = vector_to_labels(u_hu_vector_left)
#u_hu_label_right = vector_to_labels(u_hu_vector_right)

#HU_cluster_left = len(np.unique(u_hu_label_right))
#print('the cluster Hu method found (using right): ', HU_cluster_left)

#modu_hu_original_left = skn.clustering.modularity(W_HU,u_hu_label_left,resolution=0.5)
#ARI_hu_original_left = adjusted_rand_score(u_hu_label_left, gt_labels)
#purify_hu_original_left = purity_score(gt_labels, u_hu_label_left)
#inverse_purify_hu_original_left = inverse_purity_score(gt_labels, u_hu_label_left)
#NMI_hu_original_left = normalized_mutual_info_score(gt_labels, u_hu_label_left)

#print('modularity score for HU method (using left): ', modu_hu_original_left)
#print('ARI for HU method: ', ARI_hu_original_left)
#print('purify for HU method: ', purify_hu_original_left)
#print('inverse purify for HU method: ', inverse_purify_hu_original_left)
#print('NMI for HU method: ', NMI_hu_original_left)


#modu_hu_original_right = skn.clustering.modularity(W,u_hu_label_right,resolution=0.5)
#ARI_hu_original_right = adjusted_rand_score(u_hu_label_right, gt_labels)
#purify_hu_original_right = purity_score(gt_labels, u_hu_label_right)
#inverse_purify_hu_original_right = inverse_purity_score(gt_labels, u_hu_label_right)
#NMI_hu_original_right = normalized_mutual_info_score(gt_labels, u_hu_label_right)

#print('modularity score for HU method (using right): ', modu_hu_original_right)
#print('ARI for HU method: ', ARI_hu_original_right)
#print('purify for HU method: ', purify_hu_original_right)
#print('inverse purify for HU method: ', inverse_purify_hu_original_right)
#print('NMI for HU method: ', NMI_hu_original_right)



num_communities = list(range(120, 131))
m_range = list(range(5, 21))
modularity_MMBO_projection_sym_list =[]
modularity_MMBO_inner_sym_list = []
modularity_hu_sym_list=[]

# Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
#start_time_MMBO_for_loop = time.time()


#for i in m_range:
#    print('number of eigenvalues to use: ', i)
#    m=i
#    start_time_initialize = time.time()
#    u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=i)
#    time_initialize_u = time.time() - start_time_initialize
#    print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))

#    D_hu_2 = np.squeeze(eigenvalues_hu_2[:m])
#    V_hu = eigenvectors_hu_2[:,:m]

#    D_mmbo_sym = np.squeeze(eigenvalues_mmbo_sym[:m])
#    V_mmbo_sym = eigenvectors_mmbo_sym[:,:m]

#    u_hu_vector, num_iter_HU, HU_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W_HU, dt_inner, u_init,
#                             D_hu_2, V_hu, tol,inner_step_count, W_HU)

#    u_hu_label_1 = vector_to_labels(u_hu_vector)
#    modu_hu_original_1 = skn.clustering.modularity(W_HU,u_hu_label_1,resolution=0.5)
#    modularity_hu_sym_list.append(modu_hu_original_1)
#    print('modularity score for HU method: ', modu_hu_original_1)


u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1, MMBO_projection_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree_W_rw, u_init, 
                                        D_mmbo_sym, V_mmbo_sym, tol, W_rw)

u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
modularity_1_nor_lf_qh = skn.clustering.modularity(W_rw,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
modularity_MMBO_projection_sym_list.append(modularity_1_nor_lf_qh)
print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)

u_inner_nor_1,num_repeat_inner_nor, MMBO_inner_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree_W_rw, dt_inner, u_init, 
                                    D_mmbo_sym, V_mmbo_sym, tol, inner_step_count, W_rw)

u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
modularity_1_inner_nor_1 = skn.clustering.modularity(W_rw,u_inner_nor_label_1,resolution=0.5)
modularity_MMBO_inner_sym_list.append(modularity_1_inner_nor_1)
print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_1)

#num_communities = 125
#iteration_proxy = num_communities
#MMBO_projection_cluster =10
#u_proxy = u_init.copy()
#u_1_nor_Lf_Qh_individual_1=[]

#start_time_1_nor_Lf_Qh_1 = time.time()
#while num_communities > MMBO_projection_cluster:
#    num_communities = iteration_proxy
#    u_1_nor_Lf_Qh_individual_1 = u_proxy
    
    #m = iteration_proxy
    
    #D_mmbo_1 = np.squeeze(eigenvalues_mmbo_sym[:m])
    #V_mmbo_1 = eigenvectors_mmbo_sym[:,:m]

u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1, MMBO_projection_modularity_list_B = mbo_modularity_1(num_nodes,num_communities, m, degree_W_B, u_init, 
                                            D_mmbo_B_sym, V_mmbo_B_sym, tol, W_B)
    #time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
    #print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
#print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)

u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

#    MMBO_projection_cluster_list = np.unique(u_1_nor_Lf_Qh_individual_label_1)
#    print('MMBO_projection_cluster list: ', MMBO_projection_cluster_list)
#    MMBO_projection_cluster = len(MMBO_projection_cluster_list)
#    print('the cluster MMBO using projection found: ',MMBO_projection_cluster)

modularity_1_nor_lf_qh = skn.clustering.modularity(W_rw,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
#    ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_labels_rw)
#    purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1)
#    inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1)
#    NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1)

print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)
#    print('ARI for MMBO using projection with L_{mix}: ', ARI_mbo_1_nor_Lf_Qh_1)
#    print('purify for MMBO using projection with L_{mix}: ', purify_mbo_1_nor_Lf_Qh_1)
#    print('inverse purify for MMBO using projection with L_{mix}: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
#    print('NMI for MMBO using projection with L_{mix}: ', NMI_mbo_1_nor_Lf_Qh_1)
    
#    iteration_proxy = MMBO_projection_cluster
#    u_proxy = u_1_nor_Lf_Qh_individual_1


#u_1_nor_Lf_Qh_individual_1_rw,num_repeat_1_nor_Lf_Qh_1_rw, MMBO_projection_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree_W_rw, u_init, 
#                                            D_mmbo_rw, V_mmbo_rw, tol, W_rw)
#print('the number of MBO iteration for MMBO using projection with L_{mix} (rw): ', num_repeat_1_nor_Lf_Qh_1_rw)

#u_1_nor_Lf_Qh_individual_label_1_rw = vector_to_labels(u_1_nor_Lf_Qh_individual_1_rw)

#MMBO_projection_cluster_list_rw = np.unique(u_1_nor_Lf_Qh_individual_label_1_rw)
#print('MMBO_projection_cluster list: ', MMBO_projection_cluster_list)
#MMBO_projection_cluster_left = len(MMBO_projection_cluster_list_rw)
#print('the cluster MMBO using projection found (rw): ', MMBO_projection_cluster_left)

#modularity_1_nor_lf_qh_rw = skn.clustering.modularity(W_rw,u_1_nor_Lf_Qh_individual_label_1_rw,resolution=0.5)
#ARI_mbo_1_nor_Lf_Qh_rw = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1_rw, gt_labels_rw)
#purify_mbo_1_nor_Lf_Qh_rw = purity_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1_rw)
#inverse_purify_mbo_1_nor_Lf_Qh_rw = inverse_purity_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1_rw)
#NMI_mbo_1_nor_Lf_Qh_rw = normalized_mutual_info_score(gt_labels_rw, u_1_nor_Lf_Qh_individual_label_1_rw)

#print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh_rw)
#print('ARI for MMBO using projection with L_{mix}: ', ARI_mbo_1_nor_Lf_Qh_rw)
#print('purify for MMBO using projection with L_{mix}: ', purify_mbo_1_nor_Lf_Qh_rw)
#print('inverse purify for MMBO using projection with L_{mix}: ', inverse_purify_mbo_1_nor_Lf_Qh_rw)
#print('NMI for MMBO using projection with L_{mix}: ', NMI_mbo_1_nor_Lf_Qh_rw)


# MMBO1 with inner step & sym normalized L_F & Q_H
#start_time_1_inner_nor_1 = time.time()
u_inner_nor_1,num_repeat_inner_nor, MMBO_inner_modularity_list_B = mbo_modularity_inner_step(num_nodes, num_communities, m,degree_W_B, dt_inner, u_init, 
                                        D_mmbo_B_sym, V_mmbo_B_sym, tol, inner_step_count, W_B)
#time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
#print("MMBO using inner step with L_{mix}:-- %.3f seconds --" % ( time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
#print('the number of MBO iteration for MMBO using inner step with L_{mix}: ',num_repeat_inner_nor)

u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
#MMBO_finite_cluster = len(np.unique(u_inner_nor_label_1))
#print('the cluster MMBO using finite difference found: ', MMBO_finite_cluster)

modularity_1_inner_nor_1 = skn.clustering.modularity(W_rw,u_inner_nor_label_1,resolution=0.5)
#ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_labels_rw)
#purify_mbo_1_inner_nor_1 = purity_score(gt_labels_rw, u_inner_nor_label_1)
#inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_labels_rw, u_inner_nor_label_1)
#NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_labels_rw, u_inner_nor_label_1)

print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_1)
#print('ARI for MMBO using inner step with L_{mix}: ', ARI_mbo_1_inner_nor_1)
#print('purify for MMBO using inner step with L_{mix}: ', purify_mbo_1_inner_nor_1)
#print('inverse purify for MMBO using inner step with L_{mix}: ', inverse_purify_mbo_1_inner_nor_1)
#print('NMI for MMBO using inner step with L_{mix}: ', NMI_mbo_1_inner_nor_1)


#start_time_1_inner_nor_1 = time.time()
#u_inner_nor_rw,num_repeat_inner_rw = mbo_modularity_inner_step(num_nodes, num_communities, m, dt_inner, u_init, 
#                                        D_mmbo_rw, V_mmbo_rw, tol, inner_step_count)
#time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
#print("MMBO using inner step with L_{mix}:-- %.3f seconds --" % ( time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
#print('the number of MBO iteration for MMBO using inner step with L_{mix} (rw): ',num_repeat_inner_rw)

#u_inner_nor_label_rw = vector_to_labels(u_inner_nor_rw)
#MMBO_finite_cluster_rw = len(np.unique(u_inner_nor_label_rw))
#print('the cluster MMBO using finite difference found: ', MMBO_finite_cluster_rw)

#modularity_1_inner_nor_rw = skn.clustering.modularity(W_rw,u_inner_nor_label_rw,resolution=0.5)
#ARI_mbo_1_inner_nor_rw = adjusted_rand_score(u_inner_nor_label_rw, gt_labels_rw)
#purify_mbo_1_inner_nor_rw = purity_score(gt_labels_rw, u_inner_nor_label_rw)
#inverse_purify_mbo_1_inner_nor_rw = inverse_purity_score(gt_labels_rw, u_inner_nor_label_rw)
#NMI_mbo_1_inner_nor_rw = normalized_mutual_info_score(gt_labels_rw, u_inner_nor_label_rw)

#print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_rw)
#print('ARI for MMBO using inner step with L_{mix}: ', ARI_mbo_1_inner_nor_rw)
#print('purify for MMBO using inner step with L_{mix}: ', purify_mbo_1_inner_nor_rw)
#print('inverse purify for MMBO using inner step with L_{mix}: ', inverse_purify_mbo_1_inner_nor_rw)
#print('NMI for MMBO using inner step with L_{mix}: ', NMI_mbo_1_inner_nor_rw)



# Spectral clustering with k-means
#start_time_spectral_clustering = time.time()
#sc = SpectralClustering(n_clusters=10, affinity='precomputed')
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

# CNM algorithm (can setting resolution gamma)
#partition_CNM = nx_comm.greedy_modularity_communities(G)
#partition_CNM_list = [list(x) for x in partition_CNM]

#partition_CNM_expand = sum(partition_CNM_list, [])

#num_cluster_CNM = []
#for cluster in range(len(partition_CNM_list)):
#    for number_CNM in range(len(partition_CNM_list[cluster])):
#        num_cluster_CNM.append(cluster)

#print(partition_CNM_list)
#CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
#print('CNM: ',CNM_dict)
#CNM_list = list(dict.values(CNM_dict))    #convert a dict to list
#CNM_array = np.asarray(CNM_list)

#modularity_CNM = skn.clustering.modularity(W,CNM_array,resolution=0.5)
#ARI_CNM = adjusted_rand_score(CNM_array, sample_labels)
#purify_CNM = purity_score(sample_labels, CNM_array)
#inverse_purify_CNM = inverse_purity_score(sample_labels, CNM_array)
#NMI_CNM = normalized_mutual_info_score(sample_labels, CNM_array)

#print('modularity CNM score: ', modularity_CNM)
#print('ARI CNM score: ', ARI_CNM)
#print('purify for CNM: ', purify_CNM)
#print('inverse purify for CNM: ', inverse_purify_CNM)
#print('NMI for CNM: ', NMI_CNM)

# Girvan-Newman algorithm
#partition_GN = nx_comm.girvan_newman(G)
#partition_GN_list = []
#for i in next(partition_GN):
#  partition_GN_list.append(list(i))

#modularity_GN = skn.clustering.modularity(W,partition_GN_list,resolution=0.5)
#ARI_GN = adjusted_rand_score(partition_GN_list, sample_labels)
#purify_GN = purity_score(sample_labels, partition_GN_list)
#inverse_purify_GN = inverse_purity_score(sample_labels, partition_GN_list)
#NMI_GN = normalized_mutual_info_score(sample_labels, partition_GN_list)

#print('modularity CNM score: ', modularity_GN)
#print('ARI CNM score: ', ARI_GN)
#print('purify for CNM: ', purify_GN)
#print('inverse purify for CNM: ', inverse_purify_GN)
#print('NMI for CNM: ', NMI_GN)



#plt.plot(D_hu_2, ':',color='C1', label = "$L_{W_{sym}}$")
#plt.plot(D_hu_2, ':',color='C2', label = "$L_{W_{rw}}$")
#plt.plot(D_mmbo_sym, '--',color='C3', label = "$L_{W_{sym}}$ and $Q_{P_{sym}}$")
#plt.plot(D_mmbo_rw, '--',color='C4', label = "$L_{W_{rw}}$ and $Q_{P_{rw}}$")
#plt.plot(D_mmbo_B_sym, '-.',color='C5', label = "$L_{B_{sym}^+}$ and $Q_{B_{sym}^-}$")
#plt.plot(D_mmbo_B_rw, '-.',color='C6', label = "$L_{B_{rw}^+}$ and $Q_{B_{rw}^-}$")
#plt.title('Comparison of spectra.')
#plt.xlabel('Index')
#plt.ylabel('Eigenvalue')
#plt.legend()
#plt.savefig('spectra_L_MNIST.png')
#plt.show()


numbers = []
for i in range(1, 181):
    numbers.append(i)

# plot number of iteration -- modularuty 
plt.plot(numbers, HU_modularity_list, '-',color='C1', label = "Hu's method with $L_{W_{sym}}$")
plt.plot(numbers, MMBO_projection_modularity_list, '--',color='C2', label = "MMBO using projection with $L_{W_{sym}},Q_{P_{sym}}$")
plt.plot(numbers, MMBO_projection_modularity_list_B, '--',color='C3', label = "MMBO using projection with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
plt.plot(numbers, MMBO_inner_modularity_list, ':',color='C4', label = "MMBO using finite difference with $L_{W_{sym}},Q_{P_{sym}}$")
plt.plot(numbers, MMBO_inner_modularity_list_B, ':',color='C5', label = "MMBO using finite difference with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
plt.title('Modularity Score.')
plt.xlabel('Number of iterations')
plt.ylabel('Modularity')
plt.legend()
plt.savefig('Modularity_MNIST.png')
plt.show()


# plot m -- modularity
#plt.plot(numbers, modularity_hu_sym_list, '-',color='C1', label = "Hu's method with $L_{sym}$")
#plt.plot(numbers, modularity_MMBO_projection_sym_list, '--',color='C2', label = "MMBO using projection with $L_{mix_{s}}$")
#plt.plot(numbers, modularity_MMBO_inner_sym_list, ':',color='C3', label = "MMBO using finite difference with $L_{mix_{s}}$")
#plt.title('Modularity with the different chioce of $m$.')
#plt.xlabel('Number of eigenvalues to used')
#plt.ylabel('Modularity')
#plt.legend()
#plt.savefig('m-Modularity_SBM.png')
#plt.show()