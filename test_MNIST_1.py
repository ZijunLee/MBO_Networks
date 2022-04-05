import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sknetwork as skn
plt.style.use('ggplot')
import os,sys
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
from sklearn.svm import LinearSVC
import time
from graph_cut.data.read_mnist import Read_mnist_function, subsample
from graph_cut_util import build_affinity_matrix_new
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,construct_null_model
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from community import community_louvain
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
import quimb


## parameter setting
dt_inner = 1
num_communities = 10
m = 1 * num_communities
dt = 0.5
tol = 1e-5
inner_step_count =3
gamma_1 =1

gpath = '/'.join(os.getcwd().split('/')[:-1])

full_data, full_labels = Read_mnist_function(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
#full_data, full_labels = Read_mnist_function(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
full_data = full_data/255.

sample_data,sample_labels = subsample(sample_num = 1500, rd = full_data, labels = full_labels)

pca = PCA(n_components = 50, svd_solver='full')
pca.fit_transform(full_data)
train_data = pca.transform(sample_data)

del full_data

n1, p = train_data.shape
print("Features:", p)

gamma = 1. / p

feature_map_nystroem = Nystroem(gamma=gamma,random_state=1,n_components=500)
Z_training = feature_map_nystroem.fit_transform(train_data)

del train_data

adj_mat = build_affinity_matrix_new(Z_training,gamma=gamma, affinity='rbf',n_neighbors=10, neighbor_type='knearest')
print('adj_mat shape: ', adj_mat.shape)

del Z_training

null_model = construct_null_model(adj_mat)

num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

del null_model

print('symmetric normalized L_F shape: ', sym_graph_lap.shape)
print('symmetric normalized Q_H shape: ', sym_signless_lap.shape)
# Compute L_{mix} = L_{F_sym} + Q_{H_sym}
start_time_l_mix = time.time()
l_mix = sym_graph_lap + sym_signless_lap
time_l_mix = time.time() - start_time_l_mix
print("compute l_{mix}:-- %.3f seconds --" % (time_l_mix))


# Compute eigenvalues and eigenvectors of L_{mix} for MMBO
start_time_eigendecomposition_l_mix = time.time()
eigenpair_mmbo = quimb.linalg.slepc_linalg.eigs_slepc(l_mix, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
time_eig_l_mix = time.time() - start_time_eigendecomposition_l_mix
print("compute eigenvalues and eigenvectors of L_{mix} for MMBO:-- %.3f seconds --" % (time_eig_l_mix))
D_mmbo = eigenpair_mmbo[0]
V_mmbo = eigenpair_mmbo[1]


# Compute eigenvalues and eigenvectors of L_{F_sym} for HU's method
start_time_eigendecomposition_l_sym = time.time()
eigenpair_hu = quimb.linalg.slepc_linalg.eigs_slepc(sym_graph_lap, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
time_eig_l_sym = time.time() - start_time_eigendecomposition_l_sym
print("compute eigenvalues and eigenvectors of L_{F_sym} for HU's method:-- %.3f seconds --" % (time_eig_l_sym))
D_hu = eigenpair_hu[0]
V_hu = eigenpair_hu[1]


# Initialize u
start_time_initialize = time.time()
u_init = get_initial_state_1(num_nodes, num_communities, target_size)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))



# Test HU original MBO with symmetric normalized L_F
start_time_hu_original = time.time()
u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m_1,degree, dt_inner, u_init,sym_graph_lap,
                             D_hu, V_hu, tol,target_size,inner_step_count) 
time_hu_mbo = time.time() - start_time_hu_original
print("HU original MBO:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
print('HU original MBO the num_iteration: ', num_iter_HU)

u_hu_label_1 = vector_to_labels(u_hu_vector)

modu_hu_original_1 = skn.clustering.modularity(adj_mat,u_hu_label_1,resolution=0.5)
ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, sample_labels)
purify_hu_original_1 = purity_score(sample_labels, u_hu_label_1)
inverse_purify_hu_original_1 = inverse_purity_score(sample_labels, u_hu_label_1)
NMI_hu_original_1 = normalized_mutual_info_score(sample_labels, u_hu_label_1)

print(' modularity score for HU original MBO: ', modu_hu_original_1)
print(' ARI for HU original MBO: ', ARI_hu_original_1)
print(' purify for HU original MBO : ', purify_hu_original_1)
print(' inverse purify for HU original MBO : ', inverse_purify_hu_original_1)
print(' NMI for HU original MBO : ', NMI_hu_original_1)


## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
start_time_1_nor_Lf_Qh_1 = time.time()
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m_1, dt, u_init, 
                                                 l_mix, D_mmbo, V_mmbo, tol, target_size, gamma_1)
time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1                                                
print("MMBO using projection with sym normalized L_F & Q_H (K=10, m=K):-- %.3f seconds --" % (time_l_mix + time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
print('u_1 nor L_F & Q_H number of iteration(K=10 and m=K): ', num_repeat_1_nor_Lf_Qh_1)

u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)


modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, sample_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)

print(' modularity_1 normalized L_F & Q_H score(K=10 and m=K): ', modularity_1_nor_lf_qh)
print('average ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print(' purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print(' inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print(' NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)



# MMBO1 with inner step & sym normalized L_F & Q_H
start_time_1_inner_nor_1 = time.time()
u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes, num_communities, m_1, u_init, 
                                        l_mix, D_mmbo, V_mmbo, dt_inner, tol,target_size, inner_step_count)
time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
print("MMBO1 with inner step & sym normalized L_F & Q_H:-- %.3f seconds --" % (time_l_mix + time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
print('MMBO1 with inner step & sym the num_repeat_inner_nor: ',num_repeat_inner_nor)

u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)

modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat,u_inner_nor_label_1,resolution=0.5)
ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, sample_labels)
purify_mbo_1_inner_nor_1 = purity_score(sample_labels, u_inner_nor_label_1)
inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(sample_labels, u_inner_nor_label_1)
NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(sample_labels, u_inner_nor_label_1)

print(' modularity_1 inner step sym normalized score: ', modularity_1_inner_nor_1)
print(' ARI_1 inner step sym normalized score: ', ARI_mbo_1_inner_nor_1)
print(' purify for MMBO1 inner step with sym normalized \eta =1 : ', purify_mbo_1_inner_nor_1)
print(' inverse purify for MMBO1 inner step with sym normalized \eta =1 : ', inverse_purify_mbo_1_inner_nor_1)
print(' NMI for MMBO1 inner step with sym normalized \eta =1 : ', NMI_mbo_1_inner_nor_1)


# Louvain
start_time_louvain = time.time()
G = nx.convert_matrix.from_numpy_array(adj_mat)
partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)


modularity_louvain = skn.clustering.modularity(adj_mat,louvain_array,resolution=1)
ARI_louvain = adjusted_rand_score(louvain_array, sample_labels)
purify_louvain = purity_score(sample_labels, louvain_array)
inverse_purify_louvain = inverse_purity_score(sample_labels, louvain_array)
NMI_louvain = normalized_mutual_info_score(sample_labels, louvain_array)

print(' modularity Louvain score: ', modularity_louvain)
print(' ARI Louvain  score: ', ARI_louvain)
print(' purify for Louvain : ', purify_louvain)
print(' inverse purify for Louvain : ', inverse_purify_louvain)
print(' NMI for Louvain  : ', NMI_louvain)



# Spectral clustering with k-means
start_time_spectral_clustering = time.time()
sc = SpectralClustering(n_clusters=10, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)
print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)


modularity_spectral_clustering = skn.clustering.modularity(adj_mat,assignment,resolution=0.5)
ARI_spectral_clustering = adjusted_rand_score(assignment, sample_labels)
purify_spectral_clustering = purity_score(sample_labels, assignment)
inverse_purify_spectral_clustering = inverse_purity_score(sample_labels, assignment)
NMI_spectral_clustering = normalized_mutual_info_score(sample_labels, assignment)


print(' modularity Spectral clustering score(K=10 and m=K): ', modularity_spectral_clustering)
print(' ARI Spectral clustering  score: ', ARI_spectral_clustering)
print(' purify for Spectral clustering : ', purify_spectral_clustering)
print(' inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
print(' NMI for Spectral clustering: ', NMI_spectral_clustering)

