import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, mbo_modularity_2, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
from MBO_Network import mbo_modularity_1_normalized_lf,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh
from graph_mbo.utils import spectral_clustering,vector_to_labels, get_modularity_original,get_modularity,labels_to_vector,label_to_dict, purity_score,inverse_purity_score
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
import time
import csv


#Load labels, knndata, and build 10-nearest neighbor weight matrix
W = gl.weightmatrix.knn('mnist', 10, metric='vae')
#print(type(W))

gt_labels = gl.datasets.load('mnist', labels_only=True)
gt_list = gt_labels.tolist()  
#print('gt shape: ', type(gt_list))

# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_list)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_list))     # gt_label_dict is a dict


G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#print(type(G))

## parameter setting
dt_inner = 0.5
num_communities = 11
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

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(W,num_communities,m,eta_1,target_size=None)
num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, random_walk_nor_lap_06, signless_laplacian_null_model_06, nor_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(W,num_communities,m,eta_06,target_size=None)
num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, random_walk_nor_lap_05, signless_laplacian_null_model_05, nor_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(W,num_communities,m,eta_05,target_size=None)


start_time = time.time()


# mmbo 1 with unnormalized L_F and gamma=1
u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)   
u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)
#print('u_1_unnormalized label: ', u_1_unnor_individual_label)


# mmbo 1 with unnormalized L_F and gamma = 0.5
#u_1_unnor_individual_05,num_repeat_1_unnor_05 = mbo_modularity_1(num_nodes_05,num_communities, m_05,degree_05, graph_laplacian_05,signless_laplacian_null_model_05, 
#                                                tol, target_size_05,eta_05, eps=1)     
#u_1_unnor_individual_label_05 = vector_to_labels(u_1_unnor_individual_05)
#u_1_unnor_individual_label_dict_05 = label_to_dict(u_1_unnor_individual_label_05)


# mmbo 1 with unnormalized L_F and gamma = 0.6
#u_1_unnor_individual_06,num_repeat_1_unnor_06 = mbo_modularity_1(num_nodes_06,num_communities, m_06,degree_06, graph_laplacian_06,signless_laplacian_null_model_06, 
#                                                tol, target_size_06,eta_06, eps=1)     
#u_1_unnor_individual_label_06 = vector_to_labels(u_1_unnor_individual_06)
#u_1_unnor_individual_label_dict_06 = label_to_dict(u_1_unnor_individual_label_06)


# MMBO1 with normalized L_F and gamma=1
u_1_nor_individual_1,num_repeat_1_nor = mbo_modularity_1_normalized_lf(num_nodes_1,num_communities, m_1,degree_1, random_walk_nor_lap_1,signless_laplacian_null_model_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_individual_label_1 = vector_to_labels(u_1_nor_individual_1)
u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)


# MMBO1 with normalized L_F and gamma=0.5
#u_1_nor_individual_05,num_repeat_1_nor_05 = mbo_modularity_1_normalized_lf(num_nodes_05,num_communities, m_05,degree_05, nor_graph_laplacian_05,signless_laplacian_null_model_05, 
#                                                tol, target_size_05,eta_05, eps=1)       
#u_1_nor_individual_label_05 = vector_to_labels(u_1_nor_individual_05)
#u_1_nor_individual_label_dict_05 = label_to_dict(u_1_nor_individual_label_05)


# MMBO1 with normalized L_F and gamma=0.6
#u_1_nor_individual_06,num_repeat_1_nor_06 = mbo_modularity_1_normalized_lf(num_nodes_06,num_communities, m_06,degree_06, nor_graph_laplacian_06,signless_laplacian_null_model_06, 
#                                                tol, target_size_06,eta_06, eps=1)     
#u_1_nor_individual_label_06 = vector_to_labels(u_1_nor_individual_06)
#u_1_nor_individual_label_dict_06 = label_to_dict(u_1_nor_individual_label_06)


# MMBO1 with normalized Q_H and gamma=1
u_1_nor_Qh_individual_1,num_repeat_1_nor_Qh_1 = mbo_modularity_1_normalized_Qh(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_Qh_individual_label_1 = vector_to_labels(u_1_nor_Qh_individual_1)
u_1_nor_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Qh_individual_label_1)


# MMBO1 with normalized Q_H and gamma=0.6
#u_1_nor_Qh_individual_06,num_repeat_1_nor_Qh_06 = mbo_modularity_1_normalized_Qh(num_nodes_06,num_communities, m_06,degree_06, graph_laplacian_06,nor_signless_laplacian_06, 
#                                                tol, target_size_06,eta_06, eps=1)         
#u_1_nor_Qh_individual_label_06 = vector_to_labels(u_1_nor_Qh_individual_06)
#u_1_nor_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Qh_individual_label_06)


# MMBO1 with normalized Q_H and gamma=0.5
#u_1_nor_Qh_individual_05,num_repeat_1_nor_Qh_05 = mbo_modularity_1_normalized_Qh(num_nodes_05,num_communities, m_05,degree_05, graph_laplacian_05,nor_signless_laplacian_05, 
#                                                tol, target_size_05,eta_05, eps=1)   
#u_1_nor_Qh_individual_label_05 = vector_to_labels(u_1_nor_Qh_individual_05)
#u_1_nor_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Qh_individual_label_05)


# MMBO1 with normalized L_F & Q_H and gamma=1
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_1,num_communities, m_1,degree_1, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
u_1_nor_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_individual_label_1)


# MMBO1 with normalized L_F & Q_H and gamma=0.6
#u_1_nor_Lf_Qh_individual_06,num_repeat_1_nor_Lf_Qh_06 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_06,num_communities, m_06,degree_06, nor_graph_laplacian_06,nor_signless_laplacian_06, 
#                                                tol, target_size_06,eta_06, eps=1)       
#u_1_nor_Lf_Qh_individual_label_06 = vector_to_labels(u_1_nor_Lf_Qh_individual_06)
#u_1_nor_Lf_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Lf_Qh_individual_label_06)


# MMBO1 with normalized L_F & Q_H and gamma=0.5
#u_1_nor_Lf_Qh_individual_05,num_repeat_1_nor_Lf_Qh_05 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_05,num_communities, m_05,degree_05, nor_graph_laplacian_05,nor_signless_laplacian_05, 
#                                                tol, target_size_05,eta_05, eps=1)        
#u_1_nor_Lf_Qh_individual_label_05 = vector_to_labels(u_1_nor_Lf_Qh_individual_05)
#u_1_nor_Lf_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Lf_Qh_individual_label_05)



# MMBO1 with random walk L_F and gamma=1
u_1_rw_individual_1,num_repeat_1_rw = mbo_modularity_1_normalized_lf(num_nodes_1,num_communities, m_1,degree_1, random_walk_nor_lap_1,signless_laplacian_null_model_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_rw_individual_label_1 = vector_to_labels(u_1_rw_individual_1)
u_1_rw_individual_label_dict_1 = label_to_dict(u_1_rw_individual_label_1)



# MMBO1 with inner step & normalized L_F and gamma=1
u_inner_individual_1,num_repeat_inner = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, nor_graph_laplacian_1, nor_signless_laplacian_1,dt_inner, tol,target_size_1, inner_step_count)
u_inner_individual_label_1 = vector_to_labels(u_inner_individual_1)
u_inner_individual_label_dict_1 = label_to_dict(u_inner_individual_label_1)



# mmbo 2 with normalized & gamma = 1
u_2_individual_1, num_repeat_2_1 = mbo_modularity_2(num_communities, m, W, tol,eta_1,eps=1) 
u_2_individual_label_1 = vector_to_labels(u_2_individual_1)
u_2_individual_label_dict_1 = label_to_dict(u_2_individual_label_1)


# mmbo 2 with normalized & gamma = 0.6
#u_2_individual_06, num_repeat_2_06 = mbo_modularity_2(num_communities, m, W, tol,eta_06,eps=1) 
#u_2_individual_label_06 = vector_to_labels(u_2_individual_06)
#u_2_individual_label_dict_06 = label_to_dict(u_2_individual_label_06)


# mmbo 2 with normalized & gamma = 0.5
#u_2_individual_05, num_repeat_2_05 = mbo_modularity_2(num_communities, m, W, tol,eta_05,eps=1) 
#u_2_individual_label_05 = vector_to_labels(u_2_individual_05)
#u_2_individual_label_dict_05 = label_to_dict(u_2_individual_label_05)


# Louvain algorithm (can setting resolution gamma)
partition_Louvain = co.best_partition(G, resolution=1)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#print('Louvain:', type(partition_Louvain))
#print('louvain: ',louvain_list)


# CNM algorithm (can setting resolution gamma)
partition_CNM = nx_comm.greedy_modularity_communities(G)

partition_CNM_list = [list(x) for x in partition_CNM]
#print(type(partition_CNM_list))

partition_CNM_expand = sum(partition_CNM_list, [])

num_cluster_CNM = []
for cluster in range(len(partition_CNM_list)):
    for number_CNM in range(len(partition_CNM_list[cluster])):
        num_cluster_CNM.append(cluster)

#print(partition_CNM_list)
CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
#print('CNM: ',CNM_dict)

CNM_list = list(dict.values(CNM_dict))    #convert a dict to list


# Girvan-Newman algorithm
partition_GN = nx_comm.girvan_newman(G)
#print(type(partition_GN))

partition_GN_list = []
for i in next(partition_GN):
  partition_GN_list.append(list(i))
#print(partition_GN_list)

partition_GN_expand = sum(partition_GN_list, [])

num_cluster_GN = []
for cluster in range(len(partition_GN_list)):
    for number_GN in range(len(partition_GN_list[cluster])):
        num_cluster_GN.append(cluster)

#print(partition_GN_list)
GN_dict = dict(zip(partition_GN_expand, num_cluster_GN))
#print('GN: ',GN_dict)

GN_list = list(dict.values(GN_dict))    #convert a dict to list


# Spectral clustering with k-means
sc = SpectralClustering(n_clusters=10, affinity='precomputed')
assignment = sc.fit_predict(W)

ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)


#print("--- %.3f seconds ---" % (time.time() - start_time))


## Compute modularity scores

modu_gt = co.modularity(gt_label_dict,G)

modu_1_unnor_1 = co.modularity(u_1_unnor_individual_label_dict,G)
#modu_1_unnor_05 = co.modularity(u_1_unnor_individual_label_dict_05,G)
#modu_1_unnor_06 = co.modularity(u_1_unnor_individual_label_dict_06,G)

modu_1_nor_Lf_1 = co.modularity(u_1_nor_individual_label_dict_1,G)
#modu_1_nor_Lf_05 = co.modularity(u_1_nor_individual_label_dict_05,G)
#modu_1_nor_Lf_06 = co.modularity(u_1_nor_individual_label_dict_06,G)

modu_1_nor_Qh_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
#modu_1_nor_Qh_06 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_06,G)
#modu_1_nor_Qh_05 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_05,G)

modu_1_nor_Lf_Qh_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
#modu_1_nor_Lf_Qh_06 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_06,G)
#modu_1_nor_Lf_Qh_05 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_05,G)

modu_2_1 = co.modularity(u_2_individual_label_dict_1,G)
#modu_2_06 = co.modularity(u_2_individual_label_dict_06,G)
#modu_2_05 = co.modularity(u_2_individual_label_dict_05,G)

modu_inner_1 = co.modularity(u_inner_individual_label_dict_1,G)

modu_rw_1 = co.modularity(u_1_rw_individual_label_dict_1,G)

modu_louvain = co.modularity(partition_Louvain,G)
modu_CNM = co.modularity(CNM_dict,G)
modu_GN = co.modularity(GN_dict,G)
modu_sc = co.modularity(ass_dict,G)
#modularity_GN_1 = get_modularity(G,GN_dict)
#modularity_CNM_2 = nx_comm.modularity(G,partition_CNM_list)
#modu_louvain = nx_comm.modularity(G, partition_Louvain)



print('modularity_gt score:',modu_gt)
print('modularity_1 unnormalized L_F & Q_H score:',modu_1_unnor_1)

print('modularity_1 normalized L_F with \eta =1 score:',modu_1_nor_Lf_1)
#print('modularity_1 normalized L_F with \eta =0.6 score:',modu_1_nor_Lf_06)
#print('modularity_1 normalized L_F with \eta =0.5 score:',modu_1_nor_Lf_05)

print('modularity_1 normalized Q_H with \eta =1 score:',modu_1_nor_Qh_1)
#print('modularity_1 normalized Q_H with \eta =0.6 score:',modu_1_nor_Qh_06)
#print('modularity_1 normalized Q_H with \eta =0.5 score:',modu_1_nor_Qh_05)

print('modularity_1 normalized L_F & Q_H with \eta = 1 score:',modu_1_nor_Lf_Qh_1)
#print('modularity_1 normalized L_F & Q_H with \eta = 0.6 score:',modu_1_nor_Lf_Qh_06)
#print('modularity_1 normalized L_F & Q_H with \eta = 0.5 score:',modu_1_nor_Lf_Qh_05)

print('modularity_2 with \eta = 1 score:',modu_2_1)
#print('modularity_2 with \eta = 0.6 score:',modu_2_06)
#print('modularity_2 with \eta = 0.5 score:',modu_2_05)

print('modularity_inner_step score:',modu_inner_1)

print('modularity_random walk score:',modu_rw_1)

print('modularity_Louvain score:',modu_louvain)
print('modularity_CNM score:',modu_CNM)
print('modularity_GN score:',modu_GN)
#print('modularity_GN_1 score:',modularity_GN_1)
#print('modularity_CNM_2 score:',modularity_CNM_2)
print('modularity_spectral clustering score:',modu_sc)




## Compare ARI 
ARI_mbo_1_unnor_lf = adjusted_rand_score(u_1_unnor_individual_label, gt_list)
#ARI_mbo_1_unnor_lf_06 = adjusted_rand_score(u_1_unnor_individual_label_06, gt_list)
#ARI_mbo_1_unnor_lf_05 = adjusted_rand_score(u_1_unnor_individual_label_05, gt_list)

ARI_mbo_1_nor_Lf_1 = adjusted_rand_score(u_1_nor_individual_label_1, gt_list)
#ARI_mbo_1_nor_Lf_06 = adjusted_rand_score(u_1_nor_individual_label_06, gt_list)
#ARI_mbo_1_nor_Lf_05 = adjusted_rand_score(u_1_nor_individual_label_05, gt_list)

ARI_mbo_1_nor_Qh_1 = adjusted_rand_score(u_1_nor_Qh_individual_label_1, gt_list)
#ARI_mbo_1_nor_Qh_06 = adjusted_rand_score(u_1_nor_Qh_individual_label_06, gt_list)
#ARI_mbo_1_nor_Qh_05 = adjusted_rand_score(u_1_nor_Qh_individual_label_05, gt_list)

ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_list)
#ARI_mbo_1_nor_Lf_Qh_06 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_06, gt_list)
#ARI_mbo_1_nor_Lf_Qh_05 = adjusted_rand_score(u_1_nor_Qh_individual_label_05, gt_list)

ARI_mbo_2_1 = adjusted_rand_score(u_2_individual_label_1, gt_list)
#ARI_mbo_2_06 = adjusted_rand_score(u_2_individual_label_06, gt_list)
#ARI_mbo_2_05 = adjusted_rand_score(u_2_individual_label_05, gt_list)

ARI_mbo_inner_1 = adjusted_rand_score(u_inner_individual_label_1, gt_list)

ARI_mbo_rw_1 = adjusted_rand_score(u_1_rw_individual_label_1, gt_list)

ARI_spectral_clustering = adjusted_rand_score(assignment, gt_list)
ARI_GN = adjusted_rand_score(partition_GN, gt_list)
ARI_louvain = adjusted_rand_score(louvain_list, gt_list)
ARI_CNM = adjusted_rand_score(CNM_list, gt_list)



print('ARI for MMBO1 unnormalized L_F with \eta =1 : ', ARI_mbo_1_unnor_lf)
#print('ARI for MMBO1 unnormalized L_F with \eta =0.6 : ', ARI_mbo_1_unnor_lf_06)
#print('ARI for MMBO1 unnormalized L_F with \eta =0.5 : ', ARI_mbo_1_unnor_lf_05)

print('ARI for MMBO1 normalized L_F with \eta =1 : ', ARI_mbo_1_nor_Lf_1)
#print('ARI for MMBO1 normalized L_F with \eta =0.6 : ', ARI_mbo_1_nor_Lf_06)
#print('ARI for MMBO1 normalized L_F with \eta =0.5 : ', ARI_mbo_1_nor_Lf_05)

print('ARI for MMBO1 normalized Q_H with \eta =1 : ', ARI_mbo_1_nor_Qh_1)
#print('ARI for MMBO1 normalized Q_H with \eta =0.6 : ', ARI_mbo_1_nor_Qh_06)
#print('ARI for MMBO1 normalized Q_H with \eta =0.5 : ', ARI_mbo_1_nor_Qh_05)

print('ARI for MMBO1 normalized L_F & Q_H with \eta =1 : ', ARI_mbo_1_nor_Lf_Qh_1)
#print('ARI for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', ARI_mbo_1_nor_Lf_Qh_06)
#print('ARI for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', ARI_mbo_1_nor_Lf_Qh_05)

print('ARI for MMBO2 with \eta =1: ', ARI_mbo_2_1)
#print('ARI for MMBO2 with \eta =0.6: ', ARI_mbo_2_06)
#print('ARI for MMBO2 with \eta =0.5: ', ARI_mbo_2_05)

print('ARI for MBO_inner_step: ', ARI_mbo_inner_1)

print('ARI for MBO_random walk: ', ARI_mbo_rw_1)

print('ARI for spectral clustering: ', ARI_spectral_clustering)
print('ARI for GN: ', ARI_GN)
print('ARI for Louvain: ', ARI_louvain)
print('ARI for CNM: ', ARI_CNM)


# compute purify
purify_mbo_1_unnor_lf_1 = purity_score(gt_list, u_1_unnor_individual_label)
#purify_mbo_1_unnor_lf_06 = purity_score(gt_list, u_1_unnor_individual_label_06)
#purify_mbo_1_unnor_lf_05 = purity_score(gt_list, u_1_unnor_individual_label_05)

purify_mbo_1_nor_Lf_1 = purity_score(gt_list, u_1_nor_individual_label_1)
#purify_mbo_1_nor_Lf_06 = purity_score(gt_list, u_1_nor_individual_label_06)
#purify_mbo_1_nor_Lf_05 = purity_score(gt_list, u_1_nor_individual_label_05)

purify_mbo_1_nor_Qh_1 = purity_score(gt_list, u_1_nor_Qh_individual_label_1)
#purify_mbo_1_nor_Qh_06 = purity_score(gt_list, u_1_nor_Qh_individual_label_06)
#purify_mbo_1_nor_Qh_05 = purity_score(gt_list, u_1_nor_Qh_individual_label_05)

purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
#purify_mbo_1_nor_Lf_Qh_06 = purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_06)
#purify_mbo_1_nor_Lf_Qh_05 = purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_05)

purify_mbo_inner = purity_score(gt_list, u_inner_individual_label_1)

purify_mbo_rw_1 = purity_score(gt_list, u_1_rw_individual_label_1)

purify_mbo_2_1 = purity_score(gt_list, u_2_individual_label_1)
#purify_mbo_2_06 = purity_score(gt_list, u_2_individual_label_06)
#purify_mbo_2_05 = purity_score(gt_list, u_2_individual_label_05)

purify_spectral_clustering = purity_score(gt_list, assignment)
purify_gn = purity_score(gt_list, partition_GN)
purify_louvain = purity_score(gt_list, louvain_list)
purify_CNM = purity_score(gt_list, CNM_list)


print('purify for MMBO1 unnormalized L_F with \eta =1 : ', purify_mbo_1_unnor_lf_1)
#print('purify for MMBO1 unnormalized L_F with \eta =0.6 : ', purify_mbo_1_unnor_lf_06)
#print('purify for MMBO1 unnormalized L_F with \eta =0.5 : ', purify_mbo_1_unnor_lf_05)

print('purify for MMBO1 normalized L_F with \eta =1 : ', purify_mbo_1_nor_Lf_1)
#print('purify for MMBO1 normalized L_F with \eta =0.6 : ', purify_mbo_1_nor_Lf_06)
#print('purify for MMBO1 normalized L_F with \eta =0.5 : ', purify_mbo_1_nor_Lf_05)

print('purify for MMBO1 normalized Q_H with \eta =1 : ', purify_mbo_1_nor_Qh_1)
#print('purify for MMBO1 normalized Q_H with \eta =0.6 : ', purify_mbo_1_nor_Qh_06)
#print('purify for MMBO1 normalized Q_H with \eta =0.5 : ', purify_mbo_1_nor_Qh_05)

print('purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
#print('purify for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', purify_mbo_1_nor_Lf_Qh_06)
#print('purify for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', purify_mbo_1_nor_Lf_Qh_05)

print('purify for MBO_inner_step: ', purify_mbo_inner)

print('purify for MMBO1 ramdom walk L_F with \eta =1 : ', purify_mbo_rw_1)

print('purify for MMBO2 with \eta =1: ', purify_mbo_2_1)
#print('purify for MMBO2 with \eta =0.6: ', purify_mbo_2_06)
#print('purify for MMBO2 with \eta =0.5: ', purify_mbo_2_05)

#print('purify for MBO_inner_step: ', purify_mbo_inner)

print('purify for spectral clustering: ', purify_spectral_clustering)
print('purify for GN: ', purify_gn)
print('purify for Louvain: ', purify_louvain)
print('purify for CNM: ', purify_CNM)



# compute Inverse Purity
inverse_purify_mbo_1_unnor_lf_1 = inverse_purity_score(gt_list, u_1_unnor_individual_label)
#inverse_purify_mbo_1_unnor_lf_06 = inverse_purity_score(gt_list, u_1_unnor_individual_label_06)
#inverse_purify_mbo_1_unnor_lf_05 = inverse_purity_score(gt_list, u_1_unnor_individual_label_05)

inverse_purify_mbo_1_nor_Lf_1 = inverse_purity_score(gt_list, u_1_nor_individual_label_1)
#inverse_purify_mbo_1_nor_Lf_06 = inverse_purity_score(gt_list, u_1_nor_individual_label_06)
#inverse_purify_mbo_1_nor_Lf_05 = inverse_purity_score(gt_list, u_1_nor_individual_label_05)

inverse_purify_mbo_1_nor_Qh_1 = inverse_purity_score(gt_list, u_1_nor_Qh_individual_label_1)
#inverse_purify_mbo_1_nor_Qh_06 = inverse_purity_score(gt_list, u_1_nor_Qh_individual_label_06)
#inverse_purify_mbo_1_nor_Qh_05 = inverse_purity_score(gt_list, u_1_nor_Qh_individual_label_05)

inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
#inverse_purify_mbo_1_nor_Lf_Qh_06 = inverse_purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_06)
#inverse_purify_mbo_1_nor_Lf_Qh_05 = inverse_purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_05)

inverse_purify_mbo_inner = inverse_purity_score(gt_list, u_inner_individual_label_1)

inverse_purify_mbo_rw_1 = inverse_purity_score(gt_list, u_1_rw_individual_label_1)

inverse_purify_mbo_2_1 = inverse_purity_score(gt_list, u_2_individual_label_1)
#inverse_purify_mbo_2_06 = inverse_purity_score(gt_list, u_2_individual_label_06)
#inverse_purify_mbo_2_05 = inverse_purity_score(gt_list, u_2_individual_label_05)

inverse_purify_spectral_clustering = inverse_purity_score(gt_list, assignment)
inverse_purify_gn = inverse_purity_score(gt_list, partition_GN)
inverse_purify_louvain = inverse_purity_score(gt_list, louvain_list)
inverse_purify_CNM = inverse_purity_score(gt_list, CNM_list)


print('inverse purify for MMBO1 unnormalized L_F with \eta =1 : ', inverse_purify_mbo_1_unnor_lf_1)
#print('inverse purify for MMBO1 unnormalized L_F with \eta =0.6 : ', inverse_purify_mbo_1_unnor_lf_06)
#print('inverse purify for MMBO1 unnormalized L_F with \eta =0.5 : ', inverse_purify_mbo_1_unnor_lf_05)

print('inverse purify for MMBO1 normalized L_F with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_1)
#print('inverse purify for MMBO1 normalized L_F with \eta =0.6 : ', inverse_purify_mbo_1_nor_Lf_06)
#print('inverse purify for MMBO1 normalized L_F with \eta =0.5 : ', inverse_purify_mbo_1_nor_Lf_05)

print('inverse purify for MMBO1 normalized Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Qh_1)
#print('inverse purify for MMBO1 normalized Q_H with \eta =0.6 : ', inverse_purify_mbo_1_nor_Qh_06)
#print('inverse purify for MMBO1 normalized Q_H with \eta =0.5 : ', inverse_purify_mbo_1_nor_Qh_05)

print('inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
#print('inverse purify for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', inverse_purify_mbo_1_nor_Lf_Qh_06)
#print('inverse purify for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', inverse_purify_mbo_1_nor_Lf_Qh_05)

print('inverse purify for MBO_inner_step: ', inverse_purify_mbo_inner)

print('inverse purify for MMBO1 ramdom walk L_F with \eta =1 : ', inverse_purify_mbo_rw_1)

print('inverse purify for MMBO2 with \eta =1: ', inverse_purify_mbo_2_1)
#print('inverse purify for MMBO2 with \eta =0.6: ', inverse_purify_mbo_2_06)
#print('inverse purify for MMBO2 with \eta =0.5: ', inverse_purify_mbo_2_05)

#print('inverse purify for MBO_inner_step: ', inverse_purify_mbo_inner)

print('inverse purify for spectral clustering: ', inverse_purify_spectral_clustering)
print('inverse purify for GN: ', inverse_purify_gn)
print('inverse purify for Louvain: ', inverse_purify_louvain)
print('inverse purify for CNM: ', inverse_purify_CNM)



# compute Normalized Mutual Information (NMI)
NMI_mbo_1_unnor_lf_1 = normalized_mutual_info_score(gt_list, u_1_unnor_individual_label)
#NMI_mbo_1_unnor_lf_06 = normalized_mutual_info_score(gt_list, u_1_unnor_individual_label_06)
#NMI_mbo_1_unnor_lf_05 = normalized_mutual_info_score(gt_list, u_1_unnor_individual_label_05)

NMI_mbo_1_nor_Lf_1 = normalized_mutual_info_score(gt_list, u_1_nor_individual_label_1)
#NMI_mbo_1_nor_Lf_06 = normalized_mutual_info_score(gt_list, u_1_nor_individual_label_06)
#NMI_mbo_1_nor_Lf_05 = normalized_mutual_info_score(gt_list, u_1_nor_individual_label_05)

NMI_mbo_1_nor_Qh_1 = normalized_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_1)
#NMI_mbo_1_nor_Qh_06 = normalized_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_06)
#NMI_mbo_1_nor_Qh_05 = normalized_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_05)

NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
#NMI_mbo_1_nor_Lf_Qh_06 = normalized_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_06)
#NMI_mbo_1_nor_Lf_Qh_05 = normalized_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_05)

NMI_mbo_inner = normalized_mutual_info_score(gt_list, u_inner_individual_label_1)

NMI_mbo_rw_1 = normalized_mutual_info_score(gt_list, u_1_rw_individual_label_1)

NMI_mbo_2_1 = normalized_mutual_info_score(gt_list, u_2_individual_label_1)
#NMI_mbo_2_06 = normalized_mutual_info_score(gt_list, u_2_individual_label_06)
#NMI_mbo_2_05 = normalized_mutual_info_score(gt_list, u_2_individual_label_05)

NMI_spectral_clustering = normalized_mutual_info_score(gt_list, assignment)
NMI_gn = normalized_mutual_info_score(gt_list, partition_GN)
NMI_louvain = normalized_mutual_info_score(gt_list, louvain_list)
NMI_CNM = normalized_mutual_info_score(gt_list, CNM_list)


print('NMI for MMBO1 unnormalized L_F with \eta =1 : ', NMI_mbo_1_unnor_lf_1)
#print('NMI for MMBO1 unnormalized L_F with \eta =0.6 : ', NMI_mbo_1_unnor_lf_06)
#print('NMI for MMBO1 unnormalized L_F with \eta =0.5 : ', NMI_mbo_1_unnor_lf_05)

print('NMI for MMBO1 normalized L_F with \eta =1 : ', NMI_mbo_1_nor_Lf_1)
#print('NMI for MMBO1 normalized L_F with \eta =0.6 : ', NMI_mbo_1_nor_Lf_06)
#print('NMI for MMBO1 normalized L_F with \eta =0.5 : ', NMI_mbo_1_nor_Lf_05)

print('NMI for MMBO1 normalized Q_H with \eta =1 : ', NMI_mbo_1_nor_Qh_1)
#print('NMI for MMBO1 normalized Q_H with \eta =0.6 : ', NMI_mbo_1_nor_Qh_06)
#print('NMI for MMBO1 normalized Q_H with \eta =0.5 : ', NMI_mbo_1_nor_Qh_05)

print('NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)
#print('NMI for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', NMI_mbo_1_nor_Lf_Qh_06)
#print('NMI for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', NMI_mbo_1_nor_Lf_Qh_05)

print('NMI for MBO_inner_step: ', NMI_mbo_inner)

print('NMI for MMBO1 ramdom walk L_F with \eta =1 : ', NMI_mbo_rw_1)

print('NMI for MMBO2 with \eta =1: ', NMI_mbo_2_1)
#print('NMI for MMBO2 with \eta =0.6: ', NMI_mbo_2_06)
#print('NMI for MMBO2 with \eta =0.5: ', NMI_mbo_2_05)

print('NMI for MBO_inner_step: ', NMI_mbo_inner)

print('NMI for spectral clustering: ', NMI_spectral_clustering)
print('NMI for GN: ', NMI_gn)
print('NMI for Louvain: ', NMI_louvain)
print('NMI for CNM: ', NMI_CNM)



# compute Adjusted Mutual Information (AMI)
AMI_mbo_1_unnor_lf_1 = adjusted_mutual_info_score(gt_list, u_1_unnor_individual_label)
#AMI_mbo_1_unnor_lf_06 = adjusted_mutual_info_score(gt_list, u_1_unnor_individual_label_06)
#AMI_mbo_1_unnor_lf_05 = adjusted_mutual_info_score(gt_list, u_1_unnor_individual_label_05)

AMI_mbo_1_nor_Lf_1 = adjusted_mutual_info_score(gt_list, u_1_nor_individual_label_1)
#AMI_mbo_1_nor_Lf_06 = adjusted_mutual_info_score(gt_list, u_1_nor_individual_label_06)
#AMI_mbo_1_nor_Lf_05 = adjusted_mutual_info_score(gt_list, u_1_nor_individual_label_05)

AMI_mbo_1_nor_Qh_1 = adjusted_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_1)
#AMI_mbo_1_nor_Qh_06 = adjusted_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_06)
#AMI_mbo_1_nor_Qh_05 = adjusted_mutual_info_score(gt_list, u_1_nor_Qh_individual_label_05)

AMI_mbo_1_nor_Lf_Qh_1 = adjusted_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
#AMI_mbo_1_nor_Lf_Qh_06 = adjusted_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_06)
#AMI_mbo_1_nor_Lf_Qh_05 = adjusted_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_05)

AMI_mbo_inner = adjusted_mutual_info_score(gt_list, u_inner_individual_label_1)

AMI_mbo_rw_1 = adjusted_mutual_info_score(gt_list, u_1_rw_individual_label_1)

AMI_mbo_2_1 = adjusted_mutual_info_score(gt_list, u_2_individual_label_1)
#AMI_mbo_2_06 = adjusted_mutual_info_score(gt_list, u_2_individual_label_06)
#AMI_mbo_2_05 = adjusted_mutual_info_score(gt_list, u_2_individual_label_05)

AMI_spectral_clustering = adjusted_mutual_info_score(gt_list, assignment)
AMI_gn = adjusted_mutual_info_score(gt_list, partition_GN)
AMI_louvain = adjusted_mutual_info_score(gt_list, louvain_list)
AMI_CNM = adjusted_mutual_info_score(gt_list, CNM_list)


print('AMI for MMBO1 unnormalized L_F with \eta =1 : ', AMI_mbo_1_unnor_lf_1)
#print('AMI for MMBO1 unnormalized L_F with \eta =0.6 : ', AMI_mbo_1_unnor_lf_06)
#print('AMI for MMBO1 unnormalized L_F with \eta =0.5 : ', AMI_mbo_1_unnor_lf_05)

print('AMI for MMBO1 normalized L_F with \eta =1 : ', AMI_mbo_1_nor_Lf_1)
#print('AMI for MMBO1 normalized L_F with \eta =0.6 : ', AMI_mbo_1_nor_Lf_06)
#print('AMI for MMBO1 normalized L_F with \eta =0.5 : ', AMI_mbo_1_nor_Lf_05)

print('AMI for MMBO1 normalized Q_H with \eta =1 : ', AMI_mbo_1_nor_Qh_1)
#print('AMI for MMBO1 normalized Q_H with \eta =0.6 : ', AMI_mbo_1_nor_Qh_06)
#print('AMI for MMBO1 normalized Q_H with \eta =0.5 : ', AMI_mbo_1_nor_Qh_05)

print('AMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', AMI_mbo_1_nor_Lf_Qh_1)
#print('AMI for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', AMI_mbo_1_nor_Lf_Qh_06)
#print('AMI for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', AMI_mbo_1_nor_Lf_Qh_05)

print('AMI for MBO_inner_step: ', AMI_mbo_inner)

print('AMI for MMBO1 ramdom walk L_F with \eta =1 : ', AMI_mbo_rw_1)

print('AMI for MMBO2 with \eta =1: ', AMI_mbo_2_1)
#print('AMI for MMBO2 with \eta =0.6: ', AMI_mbo_2_06)
#print('AMI for MMBO2 with \eta =0.5: ', AMI_mbo_2_05)

print('AMI for spectral clustering: ', AMI_spectral_clustering)
print('AMI for GN: ', AMI_gn)
print('AMI for Louvain: ', AMI_louvain)
print('AMI for CNM: ', AMI_CNM)



testarray =["modularity_gt score", "modularity_1 unnormalized", "modularity_1 normalized L_F ", 
            "modularity_1 normalized Q_H ", "modularity_1 normalized L_F & Q_H ", "modularity_inner_step score", 
            "modularity_random walk score", "modularity_2 ", 
            "modularity_Louvain", "modularity_CNM", "modularity_GN", "modularity_spectral clustering",
            "ARI for MMBO1 unnormalized L_F ", "ARI for MMBO1 normalized L_F with eta =1", "ARI for MMBO1 normalized Q_H with eta =1", 
            "ARI for MMBO1 normalized L_F & Q_H ", "ARI for MBO_inner_step", "ARI for MBO_random walk", "ARI for MMBO2 with eta =1", 
            "ARI for Louvain", "ARI for CNM", "ARI for GN", "ARI for spectral clustering",
            "purify for MMBO1 unnormalized L_F", "purify for MMBO1 normalized L_F", "purify for MMBO1 normalized Q_H",
            "purify for MMBO1 normalized L_F & Q_H", "purify for MBO_inner_step", "purify for MMBO1 ramdom walk L_F", "purify for MMBO2",
            "purify for Louvain", "purify for CNM", "purify for GN", "purify for spectral clustering",
            "inverse purify for MMBO1 unnormalized L_F", "inverse purify for MMBO1 normalized L_F", "inverse purify for MMBO1 normalized Q_H",
            "inverse purify for MMBO1 normalized L_F & Q_H", "inverse purify for MBO_inner_step", "inverse purify for MMBO1 ramdom walk L_F", "inverse purify for MMBO2",
            "inverse purify for Louvain", "inverse purify for CNM", "inverse purify for GN", "inverse purify for spectral clustering",
            "NMI for MMBO1 unnormalized L_F", "NMI for MMBO1 normalized L_F", "NMI for MMBO1 normalized Q_H",
            "NMI for MMBO1 normalized L_F & Q_H", "NMI for MBO_inner_step", "NMI for MMBO1 ramdom walk L_F", "NMI for MMBO2",
            "NMI for Louvain", "NMI for CNM", "NMI for GN", "NMI for spectral clustering",
            "AMI for MMBO1 unnormalized L_F", "AMI for MMBO1 normalized L_F", "AMI for MMBO1 normalized Q_H",
            "AMI for MMBO1 normalized L_F & Q_H", "AMI for MBO_inner_step", "AMI for MMBO1 ramdom walk L_F", "AMI for MMBO2",
            "AMI for Louvain", "AMI for CNM", "AMI for GN", "AMI for spectral clustering"]

#resultarray = [modu_gt, modu_1_unnor_1, modu_1_unnor_06, modu_1_unnor_05,
#               modu_1_nor_Lf_1, modu_1_nor_Lf_06, modu_1_nor_Lf_05,
#               modu_1_nor_Qh_1, modu_1_nor_Qh_06, modu_1_nor_Qh_05,
#               modu_1_nor_Lf_Qh_1, modu_1_nor_Lf_Qh_06, modu_1_nor_Lf_Qh_05,
#               modu_2_1, modu_2_06, modu_2_05,
#               modu_louvain, modu_CNM, modu_sc,
#               ARI_mbo_1_unnor_lf,ARI_mbo_1_unnor_lf_06, ARI_mbo_1_unnor_lf_05,
#               ARI_mbo_1_nor_Lf_1, ARI_mbo_1_nor_Lf_06, ARI_mbo_1_nor_Lf_05,
#               ARI_mbo_1_nor_Qh_1, ARI_mbo_1_nor_Qh_06, ARI_mbo_1_nor_Qh_05,
#               ARI_mbo_1_nor_Lf_Qh_1, ARI_mbo_1_nor_Lf_Qh_06, ARI_mbo_1_nor_Lf_Qh_05,
#               ARI_mbo_2_1, ARI_mbo_2_06, ARI_mbo_2_05,
#               ARI_louvain, ARI_CNM, ARI_GN, ARI_spectral_clustering]

resultarray = [modu_gt, modu_1_unnor_1, modu_1_nor_Lf_1, modu_1_nor_Qh_1, 
               modu_1_nor_Lf_Qh_1, modu_inner_1, modu_rw_1, modu_2_1, 
               modu_louvain, modu_CNM, modu_GN, modu_sc,
               ARI_mbo_1_unnor_lf, ARI_mbo_1_nor_Lf_1, ARI_mbo_1_nor_Qh_1, ARI_mbo_1_nor_Lf_Qh_1, 
               ARI_mbo_inner_1, ARI_mbo_rw_1, ARI_mbo_2_1, 
               ARI_louvain, ARI_CNM, ARI_GN, ARI_spectral_clustering,
               purify_mbo_1_unnor_lf_1, purify_mbo_1_nor_Lf_1, purify_mbo_1_nor_Qh_1, purify_mbo_1_nor_Lf_Qh_1,
               purify_mbo_inner, purify_mbo_rw_1, purify_mbo_2_1,
               purify_louvain, purify_CNM, purify_gn, purify_spectral_clustering,
               inverse_purify_mbo_1_unnor_lf_1, inverse_purify_mbo_1_nor_Lf_1, inverse_purify_mbo_1_nor_Qh_1, inverse_purify_mbo_1_nor_Lf_Qh_1,
               inverse_purify_mbo_inner, inverse_purify_mbo_rw_1, inverse_purify_mbo_2_1,
               inverse_purify_louvain, inverse_purify_CNM, inverse_purify_gn, inverse_purify_spectral_clustering,
               NMI_mbo_1_unnor_lf_1, NMI_mbo_1_nor_Lf_1, NMI_mbo_1_nor_Qh_1, NMI_mbo_1_nor_Lf_Qh_1,
               NMI_mbo_inner, NMI_mbo_rw_1, NMI_mbo_2_1,
               NMI_louvain, NMI_CNM, NMI_gn, NMI_spectral_clustering,
               AMI_mbo_1_unnor_lf_1, AMI_mbo_1_nor_Lf_1, AMI_mbo_1_nor_Qh_1, AMI_mbo_1_nor_Lf_Qh_1,
               AMI_mbo_inner, AMI_mbo_rw_1, AMI_mbo_2_1,
               AMI_louvain, AMI_CNM, AMI_gn, AMI_spectral_clustering]


with open('MNIST_test.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray)
    wr.writerow(resultarray)

