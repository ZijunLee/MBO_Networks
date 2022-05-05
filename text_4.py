import os,sys, sklearn
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
from graph_cut.util.nystrom import nystrom_extension, nystrom_extension_test, nystrom_new, nystrom_QR, nystrom_QR_l_sym
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
from sknetwork.clustering import Louvain
from sklearn.kernel_approximation import Nystroem
from VNSC import SpectralNystrom, SpectralNystrom_new, SpectralNystrom_old
import graphlearning as gl



## parameter setting
dt_inner = 1
num_nodes = 69500
num_communities = 120
m = 1 * num_communities
#m = 100
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


#gt_number = []
#for i in sample_labels:
#    if i == 4:
#        gt_number.append(1)
#    elif i ==9:
#        gt_number.append(0)  
#gt_array = np.asarray(gt_number)  


#pca = PCA(n_components = 50,svd_solver='full')
#pca.fit(full_data)
#train_data = pca.transform(sample_data)

pca = PCA(n_components = 50,svd_solver='full')
Z_training = pca.fit_transform(data)
#print('Z_training shape: ', Z_training.shape)

#n1, p = X_test.shape

#gamma = 1. / p

#W = gl.weightmatrix.knn(Z_training, 10)
#W = rbf_kernel(Z_training, Z_training, gamma=gamma)
#print('W type: ', type(W))
#degree_W = np.array(np.sum(W, axis=-1)).flatten()
#adj_mat = W.toarray()




#D_hu = np.squeeze(eigenvalues_2[:m])
#V_hu = eigenvectors_2[:,:m]
#print('nystrom D_sign_2 shape: ', D_hu)
#print('nystrom V_sign_2 shape: ', V_hu.shape)


#adj_mat = build_affinity_matrix_new(train_data,affinity='rbf',gamma=gamma, n_neighbors=10, neighbor_type='knearest')
#print('dist_matrix type: ',type(dist_matrix))

#adj_mat = rbf_kernel(train_data, train_data)

#degree = np.array(np.sum(adj_mat, axis=-1)).flatten()
#null_model = construct_null_model(adj_mat)

#num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, num_communities,m ,target_size=None)
#l_mix =  sym_graph_lap + sym_signless_lap


# Initialize u
#start_time_initialize = time.time()
#u_init = get_initial_state_1(num_nodes, num_communities)
#u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=num_communities)
#time_initialize_u = time.time() - start_time_initialize
#print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


#start_time_eigendecomposition = time.time()
#D_sign, V_sign = eigsh(
#    sym_graph_lap,
#    k=m+1,
#    sigma=0,
#    v0=np.ones((laplacian_mix.shape[0], 1)),
#    which='SA')


#D_sign_signless, V_sign_signless = eigsh(
#    sym_signless_lap,
#    k=m+1,
#    sigma=0,
#    v0=np.ones((laplacian_mix.shape[0], 1)),
#    which='SA')
#print('D_sign signless shape: ', D_sign_signless)
#print('V_sign signless shape: ', V_sign_signless)

#D_sign_mix, V_sign_mix = eigsh(
#    l_mix,
#    k=m,
#    sigma=0,
#    v0=np.ones((laplacian_mix.shape[0], 1)),
#    which='SA')
#print('D_sign l_mix shape: ', D_sign_mix)
#print('V_sign l_mix shape: ', V_sign_mix)


#D_mix = D_sign + D_sign_signless
#V_mix = V_sign + V_sign_signless
#print('D_mix: ', D_mix)
#print('V_mix: ', V_mix)



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
#eigenvalues_hu, eigenvectors_hu = nystrom_new(Z_training, num_nystrom=500, gamma=gamma)
#D_hu = np.squeeze(eigenvalues_hu[:m])
#V_hu = eigenvectors_hu[:,:m]
#time_eig_l_sym = time.time() - start_time_l_sym
#print("nystrom extension in L_sym:-- %.3f seconds --" % (time_eig_l_sym))
#print('nystrom D_sign_1 shape: ', D_hu)
#print('nystrom V_sign_1 shape: ', V_hu.shape)


#start_time_l_mix = time.time()
#eigenvalues_mmbo, eigenvectors_mmbo = nystrom_extension_test(Z_training, num_nystrom=500, gamma=gamma)
#D_mmbo = np.squeeze(eigenvalues_1[1:m+1])
#V_mmbo = eigenvectors_1[:,1:m+1]
#time_eig_l_mix = time.time() - start_time_l_mix
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))
#print('nystrom D_sign_1 shape: ', D_mmbo)
#print('nystrom V_sign_1 shape: ', V_hu)


start_time_l_mix = time.time()
eigenvalues_mmbo, eigenvectors_mmbo, other_data, index = nystrom_QR(Z_training, num_nystrom=500, gamma=gamma)
D_mmbo = np.squeeze(eigenvalues_mmbo[1:m+1])
#print('nystrom_QR D_mmbo: ', D_mmbo)
V_mmbo = eigenvectors_mmbo[:,1:m+1]
time_eig_l_mix = time.time() - start_time_l_mix
print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_mix))


#start_time_l_sym = time.time()
#eigenvalues_hu, eigenvectors_hu, other_data, index = nystrom_QR_l_sym(Z_training, num_nystrom=500, gamma=gamma)
#D_hu = np.squeeze(eigenvalues_hu[:m])
#print('nystrom_QR D_hu: ', D_hu)
#V_hu = eigenvectors_hu[:,:m]
#time_eig_l_sym = time.time() - start_time_l_sym
#print("nystrom extension in L_mix:-- %.3f seconds --" % (time_eig_l_sym))


gt_labels = gt_labels[index[500:]]
W = gl.weightmatrix.knn(other_data, 10)
degree_W = np.array(np.sum(W, axis=-1)).flatten()


# Louvain
#start_time_louvain = time.time()
#G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#G = nx.convert_matrix.from_numpy_array(adj_mat)
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


start_time_initialize = time.time()
u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=num_communities)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


# Test HU original MBO with symmetric normalized L_F
#start_time_hu_original = time.time()
#u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W, dt_inner, u_init,
#                             D_hu, V_hu, tol,inner_step_count) 
#time_hu_mbo = time.time() - start_time_hu_original
#print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
#print('the num_iteration of HU method: ', num_iter_HU)

#u_hu_label_1 = vector_to_labels(u_hu_vector)

#modu_hu_original_1 = skn.clustering.modularity(W,u_hu_label_1,resolution=0.5)
#ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_labels)
#purify_hu_original_1 = purity_score(gt_labels, u_hu_label_1)
#inverse_purify_hu_original_1 = inverse_purity_score(gt_labels, u_hu_label_1)
#NMI_hu_original_1 = normalized_mutual_info_score(gt_labels, u_hu_label_1)

#print('modularity score for HU method: ', modu_hu_original_1)
#print('ARI for HU method: ', ARI_hu_original_1)
#print('purify for HU method: ', purify_hu_original_1)
#print('inverse purify for HU method: ', inverse_purify_hu_original_1)
#print('NMI for HU method: ', NMI_hu_original_1)

#num_communities = list(range(louvain_cluster-5, louvain_cluster+6))

## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
#start_time_1_nor_Lf_Qh_1 = time.time()
#for i in num_communities:
#    print('num_clusters: ',i)
#    m = i
#start_time_initialize = time.time()
#u_init = generate_initial_value_multiclass('rd', n_samples=num_nodes, n_class=num_communities)
#time_initialize_u = time.time() - start_time_initialize
#print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))

#D_hu = np.squeeze(eigenvalues_hu[:m])
#print('nystrom_QR D_hu shape: ', D_hu)
#V_hu = eigenvectors_hu[:,:m]
#u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m, degree_W, dt_inner, u_init,
#                            D_hu, V_hu, tol,inner_step_count) 
#print('the num_iteration of HU method: ', num_iter_HU)
#u_hu_label_1 = vector_to_labels(u_hu_vector)
#modu_hu_original_1 = skn.clustering.modularity(W,u_hu_label_1,resolution=0.5)
#print('modularity score for HU method: ', modu_hu_original_1)
    

#D_mmbo = np.squeeze(eigenvalues_mmbo[1:m+1])
#V_mmbo = eigenvectors_mmbo[:,1:m+1]

start_time_1_nor_Lf_Qh_1 = time.time()
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m, degree_W, u_init, 
                                            D_mmbo, V_mmbo, tol, W)
time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

#modularity_1_nor_lf_qh = skn.clustering.modularity(W,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
#print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)

#time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1                                                
#print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
#print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)

#u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

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
#start_time_1_inner_nor_1 = time.time()
#u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes, num_communities, m, dt_inner, u_init, 
#                                        D_mmbo, V_mmbo, tol, inner_step_count)
#time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
#print("MMBO using inner step with L_{mix}:-- %.3f seconds --" % ( time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
#print('the number of MBO iteration for MMBO using inner step with L_{mix}: ',num_repeat_inner_nor)

#u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)

#modularity_1_inner_nor_1 = skn.clustering.modularity(W,u_inner_nor_label_1,resolution=0.5)
#ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_labels)
#purify_mbo_1_inner_nor_1 = purity_score(gt_labels, u_inner_nor_label_1)
#inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_labels, u_inner_nor_label_1)
#NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_labels, u_inner_nor_label_1)

#print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_1)
#print('ARI for MMBO using inner step with L_{mix}: ', ARI_mbo_1_inner_nor_1)
#print('purify for MMBO using inner step with L_{mix}: ', purify_mbo_1_inner_nor_1)
#print('inverse purify for MMBO using inner step with L_{mix}: ', inverse_purify_mbo_1_inner_nor_1)
#print('NMI for MMBO using inner step with L_{mix}: ', NMI_mbo_1_inner_nor_1)





# Spectral clustering with k-means
start_time_spectral_clustering = time.time()
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