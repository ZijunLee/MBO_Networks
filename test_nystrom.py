# Preamble  here, add paths and import relevant modules
#sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from calendar import c
import os,sys, sklearn
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import scipy as sp
import time
from sklearn.kernel_approximation import Nystroem
from sklearn import datasets
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
# matplotlib inline
import graphlearning as gl
from sklearn.decomposition import PCA
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new
from sklearn.metrics import adjusted_rand_score
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from graph_mbo.utils import purity_score,inverse_purity_score,get_initial_state_1
from VNSC import SpectralNystrom
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,construct_null_model
from graph_mbo.utils import vector_to_labels, purity_score,inverse_purity_score



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
print('W type: ', type(W))
adj_mat = W.toarray()
#print('adj_mat type: ', type(adj_mat))

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

null_model = construct_null_model(adj_mat)
print('null_model shape: ', null_model.shape)


num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

del null_model


#del Z_training
start_time_eigendecomposition_l_sym = time.time()
D_hu, V_hu = SpectralNystrom(Z_training, gamma=gamma)
time_eig_l_sym = time.time() - start_time_eigendecomposition_l_sym
print("compute eigenvalues and eigenvectors of L_{F_sym} for HU's method:-- %.3f seconds --" % (time_eig_l_sym))

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

modu_hu_original_1 = skn.clustering.modularity(W,u_hu_label_1,resolution=0.5)
ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_labels)
purify_hu_original_1 = purity_score(gt_labels, u_hu_label_1)
inverse_purify_hu_original_1 = inverse_purity_score(gt_labels, u_hu_label_1)
NMI_hu_original_1 = normalized_mutual_info_score(gt_labels, u_hu_label_1)

print(' modularity score for HU original MBO: ', modu_hu_original_1)
print(' ARI for HU original MBO: ', ARI_hu_original_1)
print(' purify for HU original MBO : ', purify_hu_original_1)
print(' inverse purify for HU original MBO : ', inverse_purify_hu_original_1)
print(' NMI for HU original MBO : ', NMI_hu_original_1)
