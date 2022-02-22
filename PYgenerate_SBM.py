from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from MBO_Network import mbo_modularity_1, mbo_modularity_2,data_generator,SSBM_own,adj_to_laplacian_signless_laplacian
from MBO_Network import mbo_modularity_1_normalized_lf,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh
from graph_mbo.utils import spectral_clustering,vector_to_labels, get_modularity_original,get_modularity,labels_to_vector,label_to_dict
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import numpy as np
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from scipy.sparse.linalg import eigsh,eigs
import community as co
from igraph import Graph
import time
import csv

#sizes = [200, 200, 200,200,200]
#probs = [[0.95, 0.01, 0.01, 0.01,0.01], [0.01, 0.95, 0.01, 0.01, 0.01],[0.01,0.01, 0.95, 0.01, 0.01], [0.01, 0.01, 0.01, 0.95, 0.01],[0.01, 0.01, 0.01, 0.01, 0.95]]

sizes = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
#probs = [[0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], 
#         [0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], 
#         [0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.95]]

probs = [[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
         [0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
         [0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]]

G = nx.stochastic_block_model(sizes, probs, seed=0)
gt_membership = [G.nodes[v]['block'] for v in G.nodes()]    # gt_membership is a list
#print(type(gt_membership))


# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_membership)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_membership))     # gt_label_dict is a dict
#print(gt_label_dict)


# Returns the graph adjacency matrix as a NumPy matrix. 
adj_mat = nx.to_numpy_matrix(G)
#print(adj_mat)



# parameter setting
dt_inner = 0.1
num_communities = 10
m = 1 * num_communities
tol = 0.003
eta_1 = 1
eta_06 = 0.6
eta_05 = 0.5
eta_03 = 1.3
inner_step_count =3
sparsity = 0.5
noise = 0


start_time = time.time()


num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)
num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, random_walk_nor_lap_06, signless_laplacian_null_model_06, nor_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_06,target_size=None)
num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, random_walk_nor_lap_05, signless_laplacian_null_model_05, nor_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_05,target_size=None)


# mmbo 1 with unnormalized L_F and gamma=1
u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)   
u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)
#print('u_1_unnormalized label: ', u_1_unnor_individual_label)


# mmbo 1 with unnormalized L_F and gamma = 0.5
u_1_unnor_individual_05,num_repeat_1_unnor_05 = mbo_modularity_1(num_nodes_05,num_communities, m_05,degree_05, graph_laplacian_05,signless_laplacian_null_model_05, 
                                                tol, target_size_05,eta_05, eps=1)     
u_1_unnor_individual_label_05 = vector_to_labels(u_1_unnor_individual_05)
u_1_unnor_individual_label_dict_05 = label_to_dict(u_1_unnor_individual_label_05)


# mmbo 1 with unnormalized L_F and gamma = 0.6
u_1_unnor_individual_06,num_repeat_1_unnor_06 = mbo_modularity_1(num_nodes_06,num_communities, m_06,degree_06, graph_laplacian_06,signless_laplacian_null_model_06, 
                                                tol, target_size_06,eta_06, eps=1)     
u_1_unnor_individual_label_06 = vector_to_labels(u_1_unnor_individual_06)
u_1_unnor_individual_label_dict_06 = label_to_dict(u_1_unnor_individual_label_06)


# MMBO1 with normalized L_F and gamma=1
u_1_nor_individual_1,num_repeat_1_nor = mbo_modularity_1_normalized_lf(num_nodes_1,num_communities, m_1,degree_1, random_walk_nor_lap_1,signless_laplacian_null_model_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_individual_label_1 = vector_to_labels(u_1_nor_individual_1)
u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)


# MMBO1 with normalized L_F and gamma=0.5
u_1_nor_individual_05,num_repeat_1_nor_05 = mbo_modularity_1_normalized_lf(num_nodes_05,num_communities, m_05,degree_05, nor_graph_laplacian_05,signless_laplacian_null_model_05, 
                                                tol, target_size_05,eta_05, eps=1)       
u_1_nor_individual_label_05 = vector_to_labels(u_1_nor_individual_05)
u_1_nor_individual_label_dict_05 = label_to_dict(u_1_nor_individual_label_05)


# MMBO1 with normalized L_F and gamma=0.6
u_1_nor_individual_06,num_repeat_1_nor_06 = mbo_modularity_1_normalized_lf(num_nodes_06,num_communities, m_06,degree_06, nor_graph_laplacian_06,signless_laplacian_null_model_06, 
                                                tol, target_size_06,eta_06, eps=1)     
u_1_nor_individual_label_06 = vector_to_labels(u_1_nor_individual_06)
u_1_nor_individual_label_dict_06 = label_to_dict(u_1_nor_individual_label_06)


# MMBO1 with normalized Q_H and gamma=1
u_1_nor_Qh_individual_1,num_repeat_1_nor_Qh_1 = mbo_modularity_1_normalized_Qh(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_Qh_individual_label_1 = vector_to_labels(u_1_nor_Qh_individual_1)
#print('u_1_nor_Qh_individual_label_1: ',u_1_nor_Qh_individual_label_1)
u_1_nor_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Qh_individual_label_1)


# MMBO1 with normalized Q_H and gamma=0.6
u_1_nor_Qh_individual_06,num_repeat_1_nor_Qh_06 = mbo_modularity_1_normalized_Qh(num_nodes_06,num_communities, m_06,degree_06, graph_laplacian_06,nor_signless_laplacian_06, 
                                                tol, target_size_06,eta_06, eps=1)         
u_1_nor_Qh_individual_label_06 = vector_to_labels(u_1_nor_Qh_individual_06)
u_1_nor_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Qh_individual_label_06)


# MMBO1 with normalized Q_H and gamma=0.5
u_1_nor_Qh_individual_05,num_repeat_1_nor_Qh_05 = mbo_modularity_1_normalized_Qh(num_nodes_05,num_communities, m_05,degree_05, graph_laplacian_05,nor_signless_laplacian_05, 
                                                tol, target_size_05,eta_05, eps=1)   
u_1_nor_Qh_individual_label_05 = vector_to_labels(u_1_nor_Qh_individual_05)
u_1_nor_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Qh_individual_label_05)


# MMBO1 with normalized L_F & Q_H and gamma=1
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_1,num_communities, m_1,degree_1, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)     
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
u_1_nor_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_individual_label_1)


# MMBO1 with normalized L_F & Q_H and gamma=0.6
u_1_nor_Lf_Qh_individual_06,num_repeat_1_nor_Lf_Qh_06 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_06,num_communities, m_06,degree_06, nor_graph_laplacian_06,nor_signless_laplacian_06, 
                                                tol, target_size_06,eta_06, eps=1)       
u_1_nor_Lf_Qh_individual_label_06 = vector_to_labels(u_1_nor_Lf_Qh_individual_06)
u_1_nor_Lf_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Lf_Qh_individual_label_06)


# MMBO1 with normalized L_F & Q_H and gamma=0.5
u_1_nor_Lf_Qh_individual_05,num_repeat_1_nor_Lf_Qh_05 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_05,num_communities, m_05,degree_05, nor_graph_laplacian_05,nor_signless_laplacian_05, 
                                                tol, target_size_05,eta_05, eps=1)        
u_1_nor_Lf_Qh_individual_label_05 = vector_to_labels(u_1_nor_Lf_Qh_individual_05)
u_1_nor_Lf_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Lf_Qh_individual_label_05)


# mmbo 2 with normalized & gamma = 1
u_2_individual_1, num_repeat_2_1 = mbo_modularity_2(num_communities, m, adj_mat, tol,eta_1,eps=1) 
u_2_individual_label_1 = vector_to_labels(u_2_individual_1)
u_2_individual_label_dict_1 = label_to_dict(u_2_individual_label_1)


# mmbo 2 with normalized & gamma = 0.6
u_2_individual_06, num_repeat_2_06 = mbo_modularity_2(num_communities, m, adj_mat, tol,eta_06,eps=1) 
u_2_individual_label_06 = vector_to_labels(u_2_individual_06)
u_2_individual_label_dict_06 = label_to_dict(u_2_individual_label_06)


# mmbo 2 with normalized & gamma = 0.5
u_2_individual_05, num_repeat_2_05 = mbo_modularity_2(num_communities, m, adj_mat, tol,eta_05,eps=1) 
u_2_individual_label_05 = vector_to_labels(u_2_individual_05)
u_2_individual_label_dict_05 = label_to_dict(u_2_individual_label_05)


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
#partition_GN = nx_comm.girvan_newman(G)
#print(type(partition_GN))

#partition_GN_list = []
#for i in next(partition_GN):
#  partition_GN_list.append(list(i))
#print(partition_GN_list)

#partition_GN_expand = sum(partition_GN_list, [])

#num_cluster_GN = []
#for cluster in range(len(partition_GN_list)):
#    for number_GN in range(len(partition_GN_list[cluster])):
#        num_cluster_GN.append(cluster)

#print(partition_GN_list)
#GN_dict = dict(zip(partition_GN_expand, num_cluster_GN))
#print('GN: ',GN_dict)


# Spectral clustering with k-means
sc = SpectralClustering(n_clusters=10, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)

#D, V = eigsh(
#    nor_graph_laplacian_1,
#    k=num_communities,
#    v0=np.ones((nor_graph_laplacian_1.shape[0], 1)),
#    which="SA",)

#Vec = V[:, 1:].reshape((-1, 1))
#kmeans = KMeans(n_clusters=2).fit(Vec)
#print('kmeans: ', type(kmeans))
#kmeans.fit(Vec)
#assignment = kmeans.predict(Vec)
#assignment = kmeans.labels_
#print('spectral clustering: ',len(assignment))

ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)


#print("--- %.3f seconds ---" % (time.time() - start_time))


## Compute modularity scores

modu_gt = co.modularity(gt_label_dict,G)

modu_1_unnor_1 = co.modularity(u_1_unnor_individual_label_dict,G)
modu_1_unnor_05 = co.modularity(u_1_unnor_individual_label_dict_05,G)
modu_1_unnor_06 = co.modularity(u_1_unnor_individual_label_dict_06,G)

modu_1_nor_Lf_1 = co.modularity(u_1_nor_individual_label_dict_1,G)
modu_1_nor_Lf_05 = co.modularity(u_1_nor_individual_label_dict_05,G)
modu_1_nor_Lf_06 = co.modularity(u_1_nor_individual_label_dict_06,G)

modu_1_nor_Qh_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
modu_1_nor_Qh_06 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_06,G)
modu_1_nor_Qh_05 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_05,G)

modu_1_nor_Lf_Qh_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
modu_1_nor_Lf_Qh_06 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_06,G)
modu_1_nor_Lf_Qh_05 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_05,G)

modu_2_1 = co.modularity(u_2_individual_label_dict_1,G)
modu_2_06 = co.modularity(u_2_individual_label_dict_06,G)
modu_2_05 = co.modularity(u_2_individual_label_dict_05,G)
#modu_inner = co.modularity(u_inner_label_dict,G)
modu_louvain = co.modularity(partition_Louvain,G)
modu_CNM = co.modularity(CNM_dict,G)
#modu_GN = co.modularity(GN_dict,G)
modu_sc = co.modularity(ass_dict,G)
#modularity_GN_1 = get_modularity(G,GN_dict)
#modularity_CNM_2 = nx_comm.modularity(G,partition_CNM_list)
#modu_louvain = nx_comm.modularity(G, partition_Louvain)



print('modularity_gt score:',modu_gt)
print('modularity_1 unnormalized L_F & Q_H score:',modu_1_unnor_1)

#print('modularity_1 normalized L_F with \eta =1 score:',modu_1_nor_Lf_1)
#print('modularity_1 normalized L_F with \eta =0.6 score:',modu_1_nor_Lf_06)
#print('modularity_1 normalized L_F with \eta =0.5 score:',modu_1_nor_Lf_05)

print('modularity_1 normalized Q_H with \eta =1 score:',modu_1_nor_Qh_1)
print('modularity_1 normalized Q_H with \eta =0.6 score:',modu_1_nor_Qh_06)
print('modularity_1 normalized Q_H with \eta =0.5 score:',modu_1_nor_Qh_05)

print('modularity_1 normalized L_F & Q_H with \eta = 1 score:',modu_1_nor_Lf_Qh_1)
print('modularity_1 normalized L_F & Q_H with \eta = 0.6 score:',modu_1_nor_Lf_Qh_06)
print('modularity_1 normalized L_F & Q_H with \eta = 0.5 score:',modu_1_nor_Lf_Qh_05)

print('modularity_2 with \eta = 1 score:',modu_2_1)
print('modularity_2 with \eta = 0.6 score:',modu_2_06)
print('modularity_2 with \eta = 0.5 score:',modu_2_05)
#print('modularity_inner_step score:',modu_inner)
#print('modularity_original score:',modu_orig)
print('modularity_Louvain score:',modu_louvain)
print('modularity_CNM score:',modu_CNM)
#print('modularity_GN score:',modu_GN)
#print('modularity_GN_1 score:',modularity_GN_1)
#print('modularity_CNM_2 score:',modularity_CNM_2)
print('modularity_spectral clustering score:',modu_sc)




## Compare ARI 
ARI_mbo_1_unnor_lf = adjusted_rand_score(u_1_unnor_individual_label, gt_membership)
ARI_mbo_1_unnor_lf_06 = adjusted_rand_score(u_1_unnor_individual_label_06, gt_membership)
ARI_mbo_1_unnor_lf_05 = adjusted_rand_score(u_1_unnor_individual_label_05, gt_membership)

ARI_mbo_1_nor_Lf_1 = adjusted_rand_score(u_1_nor_individual_label_1, gt_membership)
ARI_mbo_1_nor_Lf_06 = adjusted_rand_score(u_1_nor_individual_label_06, gt_membership)
ARI_mbo_1_nor_Lf_05 = adjusted_rand_score(u_1_nor_individual_label_05, gt_membership)

ARI_mbo_1_nor_Qh_1 = adjusted_rand_score(u_1_nor_Qh_individual_label_1, gt_membership)
ARI_mbo_1_nor_Qh_06 = adjusted_rand_score(u_1_nor_Qh_individual_label_06, gt_membership)
ARI_mbo_1_nor_Qh_05 = adjusted_rand_score(u_1_nor_Qh_individual_label_05, gt_membership)

ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_membership)
ARI_mbo_1_nor_Lf_Qh_06 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_06, gt_membership)
ARI_mbo_1_nor_Lf_Qh_05 = adjusted_rand_score(u_1_nor_Qh_individual_label_05, gt_membership)

ARI_mbo_2_1 = adjusted_rand_score(u_2_individual_label_1, gt_membership)
ARI_mbo_2_06 = adjusted_rand_score(u_2_individual_label_06, gt_membership)
ARI_mbo_2_05 = adjusted_rand_score(u_2_individual_label_05, gt_membership)

#ARI_mbo_inner = adjusted_rand_score(u_inner_label, gt_number)
ARI_spectral_clustering = adjusted_rand_score(assignment, gt_membership)
#ARI_gn = adjusted_rand_score(partition_GN, gt_number)
ARI_louvain = adjusted_rand_score(louvain_list, gt_membership)
ARI_CNM = adjusted_rand_score(CNM_list, gt_membership)



print('ARI for MMBO1 unnormalized L_F with \eta =1 : ', ARI_mbo_1_unnor_lf)
print('ARI for MMBO1 unnormalized L_F with \eta =0.6 : ', ARI_mbo_1_unnor_lf_06)
print('ARI for MMBO1 unnormalized L_F with \eta =0.5 : ', ARI_mbo_1_unnor_lf_05)

#print('ARI for MMBO1 normalized L_F with \eta =1 : ', ARI_mbo_1_nor_Lf_1)
#print('ARI for MMBO1 normalized L_F with \eta =0.6 : ', ARI_mbo_1_nor_Lf_06)
#print('ARI for MMBO1 normalized L_F with \eta =0.5 : ', ARI_mbo_1_nor_Lf_05)

print('ARI for MMBO1 normalized Q_H with \eta =1 : ', ARI_mbo_1_nor_Qh_1)
print('ARI for MMBO1 normalized Q_H with \eta =0.6 : ', ARI_mbo_1_nor_Qh_06)
print('ARI for MMBO1 normalized Q_H with \eta =0.5 : ', ARI_mbo_1_nor_Qh_05)

print('ARI for MMBO1 normalized L_F & Q_H with \eta =1 : ', ARI_mbo_1_nor_Lf_Qh_1)
print('ARI for MMBO1 normalized L_F & Q_H with \eta =0.6 : ', ARI_mbo_1_nor_Lf_Qh_06)
print('ARI for MMBO1 normalized L_F & Q_H with \eta =0.5 : ', ARI_mbo_1_nor_Lf_Qh_05)

print('ARI for MMBO2 with \eta =1: ', ARI_mbo_2_1)
print('ARI for MMBO2 with \eta =0.6: ', ARI_mbo_2_06)
print('ARI for MMBO2 with \eta =0.5: ', ARI_mbo_2_05)

#print('ARI for MBO_inner_step: ', ARI_mbo_inner)

print('ARI for spectral clustering: ', ARI_spectral_clustering)
#print('ARI for GN: ', ARI_gn)
print('ARI for Louvain: ', ARI_louvain)
print('ARI for CNM: ', ARI_CNM)

testarray =["modularity_gt score", "modularity_1 unnormalized with eta=1", "modularity_1 unnormalized with eta=0.6", "modularity_1 unnormalized with eta=0.5",
            "modularity_1 normalized L_F with eta =1", "modularity_1 normalized L_F with eta =0.6", "modularity_1 normalized L_F with eta =0.5",
            "modularity_1 normalized Q_H with eta =1", "modularity_1 normalized Q_H with eta =0.6", "modularity_1 normalized Q_H with eta =0.5",
            "modularity_1 normalized L_F & Q_H with eta = 1", "modularity_1 normalized L_F & Q_H with eta = 0.6", "modularity_1 normalized L_F & Q_H with eta = 0.5",
            "modularity_2 with eta = 1", "modularity_2 with eta = 0.6", "modularity_2 with eta = 0.5",
            "modularity_Louvain", "modularity_CNM", "modularity_spectral clustering",
            "ARI for MMBO1 unnormalized L_F with eta =1", "ARI for MMBO1 unnormalized L_F with eta =0.6", "ARI for MMBO1 unnormalized L_F with eta =0.5",
            "ARI for MMBO1 normalized L_F with eta =1", "ARI for MMBO1 normalized L_F with eta =0.6", "ARI for MMBO1 normalized L_F with eta =0.5",
            "ARI for MMBO1 normalized Q_H with eta =1", "ARI for MMBO1 normalized Q_H with eta =0.6", "ARI for MMBO1 normalized Q_H with eta =0.5",
            "ARI for MMBO1 normalized L_F & Q_H with eta =1", "ARI for MMBO1 normalized L_F & Q_H with eta =0.6", "ARI for MMBO1 normalized L_F & Q_H with eta =0.5",
            "ARI for MMBO2 with eta =1", "ARI for MMBO2 with eta =0.6", "ARI for MMBO2 with eta =0.5",
            "ARI for Louvain", "ARI for CNM", "ARI for spectral clustering"]

resultarray = [modu_gt, modu_1_unnor_1, modu_1_unnor_06, modu_1_unnor_05,
               modu_1_nor_Lf_1, modu_1_nor_Lf_06, modu_1_nor_Lf_05,
               modu_1_nor_Qh_1, modu_1_nor_Qh_06, modu_1_nor_Qh_05,
               modu_1_nor_Lf_Qh_1, modu_1_nor_Lf_Qh_06, modu_1_nor_Lf_Qh_05,
               modu_2_1, modu_2_06, modu_2_05,
               modu_louvain, modu_CNM, modu_sc,
               ARI_mbo_1_unnor_lf,ARI_mbo_1_unnor_lf_06, ARI_mbo_1_unnor_lf_05,
               ARI_mbo_1_nor_Lf_1, ARI_mbo_1_nor_Lf_06, ARI_mbo_1_nor_Lf_05,
               ARI_mbo_1_nor_Qh_1, ARI_mbo_1_nor_Qh_06, ARI_mbo_1_nor_Qh_05,
               ARI_mbo_1_nor_Lf_Qh_1, ARI_mbo_1_nor_Lf_Qh_06, ARI_mbo_1_nor_Lf_Qh_05,
               ARI_mbo_2_1, ARI_mbo_2_06, ARI_mbo_2_05,
               ARI_louvain, ARI_CNM, ARI_spectral_clustering]


with open('SBM_test.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray)
    wr.writerow(resultarray)