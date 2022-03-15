import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from scipy.sparse.linalg import eigsh,eigs
import community as co
from community import community_louvain
import time
import csv
import sknetwork as skn

#Load labels, knndata, and build 10-nearest neighbor weight matrix
W = gl.weightmatrix.knn('mnist', 10, metric='vae')
#W_dense = W.todense()
#print(W_dense.shape)
#print(type(W_dense))

gt_labels = gl.datasets.load('mnist', labels_only=True)
gt_list = gt_labels.tolist()  
#print('gt shape: ', type(gt_list))

# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_list)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_list))     # gt_label_dict is a dict


#G = nx.convert_matrix.from_numpy_matrix(W_dense)
G = nx.convert_matrix.from_scipy_sparse_matrix(W)
print(type(G))

adj_mat = nx.convert_matrix.to_numpy_matrix(G)
print('adj_mat type: ', type(adj_mat))

## parameter setting
dt_inner = 1
num_communities = 11
m = 1 * num_communities
dt = 0.5
tol = 1e-5
inner_step_count =3
eta_1 =1

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1, rw_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)

start_time_1_nor_Lf_Qh_1 = time.time()
## Test MMBO 1 with normalized L_F

u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1,dt, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)
print("MMBO1 with normalized L_F & Q_H (K=11, m=K):-- %.3f seconds --" % (time.time() - start_time_1_nor_Lf_Qh_1))
print('u_1 nor L_F & Q_H number of iteration(K=11 and m=K): ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)


modularity_1_nor_lf_qh = skn.clustering.modularity(W,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_list)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
AMI_mbo_1_nor_Lf_Qh_1 = adjusted_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)

print('average modularity_1 normalized L_F & Q_H score(K=11 and m=5K): ', modularity_1_nor_lf_qh)
print('average ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print('average purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print('average inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print('average NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)
print('average AMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', AMI_mbo_1_nor_Lf_Qh_1)