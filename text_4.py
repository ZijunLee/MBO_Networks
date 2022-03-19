import os,sys, sklearn
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import numpy as np
import scipy as sp
from graph_cut.data.read_mnist import Read_mnist, subsample
#reload(util)
#reload(data)
#reload(gc)
from sklearn import datasets
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
# matplotlib inline
from sklearn.decomposition import PCA
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new
from sklearn.metrics import adjusted_rand_score
from MBO_Network import mbo_modularity_given_eig
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from graph_mbo.utils import purity_score,inverse_purity_score
import time
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set


## parameter setting
dt_inner = 1
num_communities = 10
m = 1 * num_communities
dt = 0.5
tol = 1e-5
inner_step_count =3
eta_1 =1

#gpath = '/'.join(os.getcwd().split('/')[:-1])

#raw_data, labels = Read_mnist(digits = [4,5,9],path = gpath+'/MBO_signed_graphs/graph_cut/data') 
#raw_data = raw_data/255.
#full_data, full_labels = Read_mnist(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
full_data, full_labels = Read_mnist(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
#full_data, full_labels = Read_mnist(digits = range(10))
full_data = full_data/255.
sample_data,sample_labels = subsample(sample_num = 1000, rd = full_data, labels = full_labels)
print('sample number is 1000')
print('sample_labels: ',sample_labels)

adj_mat = build_affinity_matrix_new(sample_data,affinity='z-p',gamma=1, n_neighbors=10, neighbor_type='knearest')
print('adj_mat type: ',adj_mat.shape)

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1, rw_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)

start_time_1_nor_Lf_Qh_1 = time.time()
## Test MMBO 1 with normalized L_F

u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes_1,num_communities, m_1, degree_1, nor_graph_laplacian_1,nor_signless_laplacian_1,
                                                 tol, target_size_1, eta_1)
                                                
print("MMBO1 with normalized L_F & Q_H (K=11, m=K):-- %.3f seconds --" % (time.time() - start_time_1_nor_Lf_Qh_1))
print('u_1 nor L_F & Q_H number of iteration(K=11 and m=K): ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)
print('u_1_nor_Lf_Qh_individual_label_1: ', u_1_nor_Lf_Qh_individual_label_1)


modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, sample_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
AMI_mbo_1_nor_Lf_Qh_1 = adjusted_mutual_info_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)

print('average modularity_1 normalized L_F & Q_H score(K=11 and m=K): ', modularity_1_nor_lf_qh)
print('average ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print('average purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print('average inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print('average NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)
print('average AMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', AMI_mbo_1_nor_Lf_Qh_1)