import numpy as np
import graphlearning as gl
from graph_mbo.utils import vector_to_labels,labels_to_vector,label_to_dict, purity_score,inverse_purity_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
import sknetwork as skn
from community import community_louvain
import time
import csv
import quimb
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,construct_null_model
from graph_cut_util import build_affinity_matrix_new
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1



## parameter setting
dt_inner = 1
num_communities = 10
m = 1 * num_communities
dt = 0.5
tol = 1e-5
inner_step_count =3
gamma_1 =1


#Load labels, knndata, and build 10-nearest neighbor weight matrix
#W = gl.weightmatrix.knn('mnist', 10)


data, gt_labels = gl.datasets.load('mnist')
#gt_list = gt_labels.tolist()
#print(data.shape)
#print(type(data))
print('gt shape: ', gt_labels.shape)


#pca = PCA(n_components = 50)
#train_data = pca.fit_transform(data)
#train_data = pca.transform(sample_data)
#print('train_data shape: ', type(train_data))

#del data

#n1, p = train_data.shape
#print("Features:", p)

gamma = 0.02

feature_map_nystroem = Nystroem(gamma=gamma,random_state=1,n_components=50)
Z_training = feature_map_nystroem.fit_transform(data)
#print('Z_training: ', Z_training)
#print('Z_training shape: ', Z_training.shape)

#n1, p = Z_training.shape
#print("Features:", p)

W = gl.weightmatrix.knn(Z_training, 10)
print('W shape: ', W.shape)
#print('W type: ', type(W))


#gt_labels = gl.datasets.load('mnist', labels_only=True)
#gt_list = gt_labels.tolist()  
#print('gt shape: ', type(gt_list))

# convert a list to a dict
#gt_label_dict = []
#len_gt_label = []

#for e in range(len(gt_list)):
#    len_gt_label.append(e)

#gt_label_dict = dict(zip(len_gt_label, gt_list))     # gt_label_dict is a dict


#del train_data

#adj_mat = build_affinity_matrix_new(Z_training,gamma=gamma, affinity='rbf',n_neighbors=10, neighbor_type='knearest')
#print('adj_mat shape: ', adj_mat.shape)
adj_mat = W.toarray()
#print('adj_mat type: ', type(adj_mat))

del Z_training


start_time_construct_null_model = time.time()
null_model = construct_null_model(adj_mat)
time_null_model = time.time() - start_time_construct_null_model
print("construct null model:-- %.3f seconds --" % (time_null_model))


start_time_construct_lap_signless = time.time()
num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)
time_laplacian = time.time() - start_time_construct_lap_signless
print("construct laplacian & signless laplacian:-- %.3f seconds --" % (time_laplacian))

del null_model

#print('symmetric normalized L_F shape: ', sym_graph_lap.shape)
#print('symmetric normalized Q_H shape: ', sym_signless_lap.shape)
# Compute L_{mix} = L_{F_sym} + Q_{H_sym}
start_time_l_mix = time.time()
l_mix = sym_graph_lap + sym_signless_lap
time_l_mix = time.time() - start_time_l_mix
print("compute l_{mix}:-- %.3f seconds --" % (time_l_mix))


print('Using ARPACK for eigen-decomposition')
# Compute eigenvalues and eigenvectors of L_{mix} for MMBO
start_time_eigendecomposition_l_mix = time.time()
#eigenpair_mmbo = quimb.linalg.slepc_linalg.eigs_slepc(l_mix, m, B=None,which='SA',isherm=True, return_vecs=True,EPSType='krylovschur',tol=1e-7,maxiter=10000)
D_mmbo, V_mmbo = eigsh(
    l_mix,
    k=m,
#    sigma=0,
#    v0=np.ones((laplacian_mix.shape[0], 1)),
    which='SA')
time_eig_l_mix = time.time() - start_time_eigendecomposition_l_mix
print("compute eigenvalues and eigenvectors of L_{mix} for MMBO:-- %.3f seconds --" % (time_eig_l_mix))

# Initialize u
start_time_initialize = time.time()
u_init = get_initial_state_1(num_nodes, num_communities, target_size)
time_initialize_u = time.time() - start_time_initialize
print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))



## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
start_time_1_nor_Lf_Qh_1 = time.time()
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m_1, dt, u_init, 
                                                 l_mix, D_mmbo, V_mmbo, tol, target_size, gamma_1)
time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1                                                
print("MMBO using projection with sym normalized L_F & Q_H (K=10, m=K):-- %.3f seconds --" % (time_l_mix + time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
print('u_1 nor L_F & Q_H number of iteration(K=10 and m=K): ', num_repeat_1_nor_Lf_Qh_1)

u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)


modularity_1_nor_lf_qh = skn.clustering.modularity(W,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_labels, u_1_nor_Lf_Qh_individual_label_1)

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

modularity_1_inner_nor_1 = skn.clustering.modularity(W,u_inner_nor_label_1,resolution=0.5)
ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_labels)
purify_mbo_1_inner_nor_1 = purity_score(gt_labels, u_inner_nor_label_1)
inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_labels, u_inner_nor_label_1)
NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_labels, u_inner_nor_label_1)

print(' modularity_1 inner step sym normalized score: ', modularity_1_inner_nor_1)
print(' ARI_1 inner step sym normalized score: ', ARI_mbo_1_inner_nor_1)
print(' purify for MMBO1 inner step with sym normalized \eta =1 : ', purify_mbo_1_inner_nor_1)
print(' inverse purify for MMBO1 inner step with sym normalized \eta =1 : ', inverse_purify_mbo_1_inner_nor_1)
print(' NMI for MMBO1 inner step with sym normalized \eta =1 : ', NMI_mbo_1_inner_nor_1)


# Louvain
start_time_louvain = time.time()
G = nx.convert_matrix.from_sp(adj_mat)
partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)


modularity_louvain = skn.clustering.modularity(W,louvain_array,resolution=1)
ARI_louvain = adjusted_rand_score(louvain_array, gt_labels)
purify_louvain = purity_score(gt_labels, louvain_array)
inverse_purify_louvain = inverse_purity_score(gt_labels, louvain_array)
NMI_louvain = normalized_mutual_info_score(gt_labels, louvain_array)

print(' modularity Louvain score: ', modularity_louvain)
print(' ARI Louvain  score: ', ARI_louvain)
print(' purify for Louvain : ', purify_louvain)
print(' inverse purify for Louvain : ', inverse_purify_louvain)
print(' NMI for Louvain  : ', NMI_louvain)
