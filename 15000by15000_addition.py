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
print('W type: ', type(W))


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

null_model = construct_null_model(adj_mat)
print('null_model shape: ', null_model.shape)

num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

del null_model

#print('symmetric normalized L_F shape: ', sym_graph_lap.shape)
#print('symmetric normalized Q_H shape: ', sym_signless_lap.shape)
# Compute L_{mix} = L_{F_sym} + Q_{H_sym}
start_time_l_mix = time.time()
l_mix = sym_graph_lap + sym_signless_lap
time_l_mix = time.time() - start_time_l_mix
print("compute l_{mix}:-- %.3f seconds --" % (time_l_mix))



