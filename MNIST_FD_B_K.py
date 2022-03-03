import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step,MMBO2_preliminary
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
#num_communities_10 = 11
m = 1 * num_communities
#m_1 = 2 * num_communities
#m = 3
dt = 0.5
tol = 1e-5

#tol = 0
eta_1 = 1
eta_06 = 0.6
eta_05 = 0.5
eta_03 = 1.3
inner_step_count =3



num_nodes_B_1, m_B_1, target_size_B_1, graph_laplacian_positive_B_1, sym_lap_positive_B_1, rw_nor_lap_positive_B_1, signless_lap_neg_B_1, sym_signless_lap_negative_B_1, rw_signless_lap_negative_B_1 = MMBO2_preliminary(adj_mat, num_communities,m,eta_1)



start_time_2_inner_B_unnor_1 = time.time()
# MMBO1 with inner step & unnormalized B^+ & B^- and gamma=1

u_inner_B_unnor_1,num_repeat_inner_B_unnor = mbo_modularity_inner_step(num_nodes_B_1, num_communities, m_B_1, graph_laplacian_positive_B_1, signless_lap_neg_B_1,dt_inner, tol,target_size_B_1, inner_step_count)
u_inner_B_unnor_label_1 = vector_to_labels(u_inner_B_unnor_1)
u_inner_B_unnor_label_dict_1 = label_to_dict(u_inner_B_unnor_label_1)

print("MMBO1 with inner step & unnormalized B^+ & B^- and gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_unnor_1))

modularity_1_inner_B_unnor_1 = skn.clustering.modularity(W,u_inner_B_unnor_label_1,resolution=1)
ARI_mbo_1_inner_B_unnor_1 = adjusted_rand_score(u_inner_B_unnor_label_1, gt_list)
purify_mbo_1_inner_B_unnor_1 = purity_score(gt_list, u_inner_B_unnor_label_1)
inverse_purify_mbo_1_inner_B_unnor_1 = inverse_purity_score(gt_list, u_inner_B_unnor_label_1)
NMI_mbo_1_inner_B_unnor_1 = normalized_mutual_info_score(gt_list, u_inner_B_unnor_label_1)
AMI_mbo_1_inner_B_unnor_1 = adjusted_mutual_info_score(gt_list, u_inner_B_unnor_label_1)


print(' modularity_1 inner step unnormalized B^+ & B^-score: ', modularity_1_inner_B_unnor_1)
print(' ARI_1 inner step unnormalized B^+ & B^-score: ', ARI_mbo_1_inner_B_unnor_1)
print(' purify for MMBO1 inner step withunnormalized B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_unnor_1)
print(' inverse purify for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_unnor_1)
print(' NMI for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_unnor_1)
print(' AMI for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_unnor_1)


start_time_2_inner_B_nor_1 = time.time()
# MMBO1 with inner step & sym normalized B^+ & B^- and gamma=1

u_inner_B_nor_1,num_repeat_inner_B_nor = mbo_modularity_inner_step(num_nodes_B_1, num_communities, m_B_1, sym_lap_positive_B_1, sym_signless_lap_negative_B_1,dt_inner, tol,target_size_B_1, inner_step_count)
u_inner_B_nor_label_1 = vector_to_labels(u_inner_B_nor_1)
u_inner_B_nor_label_dict_1 = label_to_dict(u_inner_B_nor_label_1)

print("MMBO1 with inner step & sym normalized B^+ & B^- gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_nor_1))

modularity_1_inner_B_nor_1 = skn.clustering.modularity(W,u_inner_B_nor_label_1,resolution=1)
ARI_mbo_1_inner_B_nor_1 = adjusted_rand_score(u_inner_B_nor_label_1, gt_list)
purify_mbo_1_inner_B_nor_1 = purity_score(gt_list, u_inner_B_nor_label_1)
inverse_purify_mbo_1_inner_B_nor_1 = inverse_purity_score(gt_list, u_inner_B_nor_label_1)
NMI_mbo_1_inner_B_nor_1 = normalized_mutual_info_score(gt_list, u_inner_B_nor_label_1)
AMI_mbo_1_inner_B_nor_1 = adjusted_mutual_info_score(gt_list, u_inner_B_nor_label_1)

print(' modularity_1 inner step sym normalized B^+ & B^- score: ', modularity_1_inner_B_nor_1)
print(' ARI_1 inner step sym normalized B^+ & B^- score: ', ARI_mbo_1_inner_B_nor_1)
print(' purify for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_nor_1)
print(' inverse purify for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_nor_1)
print(' NMI for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_nor_1)
print(' AMI for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_nor_1)


start_time_2_inner_B_rw_1 = time.time()
# MMBO1 with inner step & random ealk B^+ & B^- and gamma=1

u_inner_B_rw_1,num_repeat_inner_B_rw = mbo_modularity_inner_step(num_nodes_B_1, num_communities, m_B_1, rw_nor_lap_positive_B_1, rw_signless_lap_negative_B_1,dt_inner, tol,target_size_B_1, inner_step_count)
u_inner_B_rw_label_1 = vector_to_labels(u_inner_B_rw_1)
u_inner_B_rw_label_dict_1 = label_to_dict(u_inner_B_rw_label_1)

print("MMBO1 with inner step & random Walk B^+ & B^- gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_rw_1))

modularity_1_inner_B_rw_1 = skn.clustering.modularity(W,u_inner_B_rw_label_1,resolution=1)
ARI_mbo_1_inner_B_rw_1 = adjusted_rand_score(u_inner_B_rw_label_1, gt_list)
purify_mbo_1_inner_B_rw_1 = purity_score(gt_list, u_inner_B_rw_label_1)
inverse_purify_mbo_1_inner_B_rw_1 = inverse_purity_score(gt_list, u_inner_B_rw_label_1)
NMI_mbo_1_inner_B_rw_1 = normalized_mutual_info_score(gt_list, u_inner_B_rw_label_1)
AMI_mbo_1_inner_B_rw_1 = adjusted_mutual_info_score(gt_list, u_inner_B_rw_label_1)

print(' modularity_1 inner step random ealk B^+ & B^- score: ', modularity_1_inner_B_rw_1)
print(' ARI_1 inner step random ealk B^+ & B^- score: ', ARI_mbo_1_inner_B_rw_1)
print(' purify for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_rw_1)
print(' inverse purify for MMBO1 inner step with srandom ealk B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_rw_1)
print(' NMI for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_rw_1)
print(' AMI for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_rw_1)
