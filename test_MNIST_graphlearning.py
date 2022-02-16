import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, mbo_modularity_2
from MBO_Network import mbo_modularity_1_normalized_lf,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh
from graph_mbo.utils import spectral_clustering,vector_to_labels, get_modularity_original,get_modularity,labels_to_vector,label_to_dict
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh,eigs
import community as co
import time

start_time = time.time()

#Load labels, knndata, and build 10-nearest neighbor weight matrix
#gt_non_label = gl.datasets.load('mnist', labels_only=False)
gt_labels = gl.datasets.load('mnist', labels_only=True)

W = gl.weightmatrix.knn('mnist', 10, metric='vae')
#print(W.shape)

G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#print(type(G))

## parameter setting
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


## Test MMBO 1 with unnormalized L_F & Q_H
u_1_unnor_1,num_repeat_1_unnor = mbo_modularity_1(num_communities,m, W, tol,eta_1)   
u_1_unnor_label = vector_to_labels(u_1_unnor_1)
u_1_unnor_label_dict = label_to_dict(u_1_unnor_label)
print('number of itration of MMBO 1 with unnormalized L_F & Q_H: ', num_repeat_1_unnor)
#print('u_1: ',u_1)


## Test MMBO 1 with normalized L_F and unnormalized Q_H, gamma=1
u_1_nor_1,num_repeat_1_nor = mbo_modularity_1_normalized_lf(num_communities,m, W, tol,eta_1)   
u_1_nor_label = vector_to_labels(u_1_nor_1)
u_1_nor_label_dict = label_to_dict(u_1_nor_label)

print('number of itration of MMBO 1 with normalized L_F and unnormalized Q_H: ', num_repeat_1_nor)
#print(u_1)


## MMBO1 with normalized Q_H and gamma=1
u_1_nor_Qh_1,num_repeat_1_nor_Qh_1 = mbo_modularity_1_normalized_Qh(num_communities,m, W, tol,eta_1)
print('number of itration of MMBO 1 with unnormalized L_F and normalized Q_H: ', num_repeat_1_nor_Qh_1)     
u_1_nor_Qh_label_1 = vector_to_labels(u_1_nor_Qh_1)
u_1_nor_Qh_label_dict_1 = label_to_dict(u_1_nor_Qh_label_1)


## MMBO1 with normalized L_F & Q_H and gamma=1
u_1_nor_Lf_Qh_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1_normalized_Lf_Qh(num_communities,m, W, tol,eta_1)
print('number of itration of MMBO 1 with normalized L_F and normalized Q_H: ', num_repeat_1_nor_Lf_Qh_1)      
u_1_nor_Lf_Qh_label_1 = vector_to_labels(u_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_label_1)


## Test MMBO 2

u_2,nun_times_2 = mbo_modularity_2(num_communities,m, W, tol,eta_1)
print('number of itration of MMBO 2 with normalized L_F and normalized Q_H: ', nun_times_2)  
u_2_label = vector_to_labels(u_2)
u_2_label_dict = label_to_dict(u_2_label)



# Louvain algorithm (can setting resolution gamma)
partition_Louvain = co.best_partition(G, resolution=1)

#print('Louvain:',partition_Louvain)


# CNM algorithm (can setting resolution gamma)
partition_CNM = nx_comm.greedy_modularity_communities(G)
partition_CNM_list = [list(x) for x in partition_CNM]                   

partition_CNM_expand = sum(partition_CNM_list, [])

num_cluster_CNM = []
for cluster in range(len(partition_CNM_list)):
    for number_CNM in range(len(partition_CNM_list[cluster])):
        num_cluster_CNM.append(cluster)

#print(partition_CNM_list)
CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
#print('CNM: ',CNM_dict)


# Girvan-Newman algorithm
partition_GN = nx_comm.girvan_newman(G)
#print(partition_GN)

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


# Spectral clustering with k-means
num_communities = 11

sc_degree = np.array(np.sum(W, axis=1)).flatten()
sc_num_nodes = len(sc_degree)
#graph_laplacian, degree = sp.sparse.csgraph.laplacian(W, return_diag=True)
sc_degree_diag = sp.sparse.spdiags([sc_degree], [0], sc_num_nodes, sc_num_nodes)
sc_graph_laplacian = sc_degree_diag - W
sc_degree_inv = sp.sparse.spdiags([1.0 / sc_degree], [0], sc_num_nodes, sc_num_nodes)
sym_graph_laplacian = np.sqrt(sc_degree_inv) @ sc_graph_laplacian @ np.sqrt(sc_degree_inv)
D, V = eigsh(
    sym_graph_laplacian,
    k=num_communities,
    v0=np.ones((sym_graph_laplacian.shape[0], 1)),
    which="SA",)
V = V[:, 1:].reshape((-1, 1))
kmeans = KMeans(n_clusters=2)
kmeans.fit(V)
assignment = kmeans.predict(V)
#print(assignment)
ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)


## Compute modularity scores

#modu_gt = co.modularity(gt_label_dict,G)
modu_mmbo1_unnor_Lf_Qh = co.modularity(u_1_unnor_label_dict,G)
modu_mmbo1_norLf_unnorQh = co.modularity(u_1_nor_label_dict,G)
modu_mmbo1_unnorLf_norQh = co.modularity(u_1_nor_Qh_label_dict_1,G)
modu_mmbo1_nor_Lf_Qh = co.modularity(u_1_nor_Lf_Qh_label_dict_1,G)

modu_mmbo2 = co.modularity(u_2_label_dict,G)

modu_louvain = co.modularity(partition_Louvain,G)
#modu_louvain = nx_comm.modularity(G, partition_Louvain, resolution =0.7)
modu_CNM = co.modularity(CNM_dict,G)
modu_GN = co.modularity(GN_dict,G)
modu_sc = co.modularity(ass_dict,G)
#modularity_GN_1 = get_modularity(G,GN_dict)
#modularity_CNM_2 = nx_comm.modularity(G,partition_CNM_list)


#print('modularity_gt score:',modu_gt)
print('modularity score of MMBO 1 with unnormalized L_F & Q_H:',modu_mmbo1_unnor_Lf_Qh)
print('modularity score of MMBO 1 with normalized L_F & unnormalized Q_H:',modu_mmbo1_norLf_unnorQh)
print('modularity score of MMBO 1 with unnormalized L_F & normalized Q_H:',modu_mmbo1_unnorLf_norQh)
print('modularity score of MMBO 1 with normalized L_F & Q_H:',modu_mmbo1_nor_Lf_Qh)
print('modularity score of normalized MMBO 2:',modu_mmbo2)
#print('modularity_original score:',modu_orig)
print('modularity_Louvain score:',modu_louvain)
print('modularity_CNM score:',modu_CNM)
print('modularity_GN score:',modu_GN)
#print('modularity_GN_1 score:',modularity_GN_1)
#print('modularity_CNM_2 score:',modularity_CNM_2)
print('modularity_spectral clustering score:',modu_sc)

#print("--- %.3f seconds ---" % (time.time() - start_time))

## Compare ARI 
ARI_mmbo1_unnor_Lf_Qh = adjusted_rand_score(u_1_unnor_label, gt_labels)
ARI_mmbo1_norLf_unnorQh = adjusted_rand_score(u_1_nor_label, gt_labels)
ARI_mmbo1_unnorLf_norQh = adjusted_rand_score(u_1_nor_Qh_label_1, gt_labels)
ARI_mmbo1_nor_Lf_Qh = adjusted_rand_score(u_1_nor_Lf_Qh_label_1, gt_labels)

ARI_mmbo_2 = adjusted_rand_score(u_2_label, gt_labels)

ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)

print('ARI for MMBO 1 with unnormalized L_F & Q_H: ', ARI_mmbo1_unnor_Lf_Qh)
print('ARI for MMBO 1 with normalized L_F & unnormalized Q_H: ', ARI_mmbo1_norLf_unnorQh)
print('ARI for MMBO 1 with unnormalized L_F & normalized Q_H: ', ARI_mmbo1_unnorLf_norQh)
print('ARI for MMBO 1 with normalized L_F & Q_H: ', ARI_mmbo1_nor_Lf_Qh)

print('ARI for normalized MMBO 2: ', ARI_mmbo_2)
#print('ARI for MBO_inner_step: ', ARI_mbo_inner)
#print('ARI for MBO_original: ', ARI_mbo_ori)
print('ARI for spectral clustering: ', ARI_spectral_clustering)