import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
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
dt_inner = 0.1
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

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1, rw_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)


start_time_1_unnor_1 = time.time()

# MMBO 1 with unnormalized L_F and gamma=1

u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)   
print('u_1 unnor L_F & Q_H number of iteration (K=12): ', num_repeat_1_unnor)
u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
#u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)
#u_1_unnor_label_set = dict_to_list_set(u_1_unnor_individual_label_dict)

print("mmbo 1 with unnormalized L_F and gamma=1 (K=12):-- %.3f seconds --" % (time.time() - start_time_1_unnor_1))

#modularity_1_unnor_individual = co.modularity(u_1_unnor_individual_label_dict,G)
#modularity_1_unnor_lf_qh = nx_comm.modularity(G,u_1_unnor_label_set)
modu_1_unnor_Lf_Qh = skn.clustering.modularity(W,u_1_unnor_individual_label,resolution=0.5)
ARI_mbo_1_unnor_lf = adjusted_rand_score(u_1_unnor_individual_label, gt_list)
purify_mbo_1_unnor_lf_1 = purity_score(gt_list, u_1_unnor_individual_label)
inverse_purify_mbo_1_unnor_lf_1 = inverse_purity_score(gt_list, u_1_unnor_individual_label)
NMI_mbo_1_unnor_lf_1 = normalized_mutual_info_score(gt_list, u_1_unnor_individual_label)
AMI_mbo_1_unnor_lf_1 = adjusted_mutual_info_score(gt_list, u_1_unnor_individual_label)

print('average modularity_1 unnormalized L_F & Q_H score(K=12): ', modu_1_unnor_Lf_Qh)
print('average ARI_1 unnormalized L_F & Q_H score: ', ARI_mbo_1_unnor_lf)
print('average purify for MMBO1 unnormalized L_F with \eta =1 : ', purify_mbo_1_unnor_lf_1)
print('average inverse purify for MMBO1 unnormalized L_F with \eta =1 : ', inverse_purify_mbo_1_unnor_lf_1)
print('average NMI for MMBO1 unnormalized L_F with \eta =1 : ', NMI_mbo_1_unnor_lf_1)
print('average AMI for MMBO1 unnormalized L_F with \eta =1 : ', AMI_mbo_1_unnor_lf_1)

testarray_unnor_Lf_Qh = ["average modularity_1 unnormalized L_F & Q_H score", "average ARI_1 unnormalized L_F & Q_H score",
             "average purify for MMBO1 unnormalized L_F", "average inverse purify for MMBO1 unnormalized L_F",
             "average NMI for MMBO1 unnormalized L_F", "average AMI for MMBO1 unnormalized L_F"]

#resultarray = [average_mbo_1_unnor, average_ARI_1_unnor,
#               average_purify_1_unnor_1, average_inverse_purify_1_unnor_1,
#               average_NMI_1_unnor_1, average_AMI_1_unnor_1]

resultarray_unnor_Lf_Qh = [modu_1_unnor_Lf_Qh, ARI_mbo_1_unnor_lf,
               purify_mbo_1_unnor_lf_1, inverse_purify_mbo_1_unnor_lf_1,
               NMI_mbo_1_unnor_lf_1, AMI_mbo_1_unnor_lf_1]

with open('MNIST_unnor_LF_QH.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_unnor_Lf_Qh)
    wr.writerow(resultarray_unnor_Lf_Qh)


start_time_1_nor_Lf_Qh_1 = time.time()

# MMBO1 with normalized L_F & Q_H and gamma=1

u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1, eps=1)     
print('u_1 nor L_F & Q_H number of iteration(K=12): ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_individual_label_1)
#u_1_nor_Lf_Qh_label_set = dict_to_list_set(u_1_nor_Lf_Qh_individual_label_dict_1)

print("MMBO1 with normalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_nor_Lf_Qh_1))

#modularity_1_nor_Lf_Qh_individual_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
#modularity_1_nor_lf_qh = nx_comm.modularity(G,u_1_nor_Lf_Qh_label_set)
modularity_1_nor_lf_qh = skn.clustering.modularity(W,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_list)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)
AMI_mbo_1_nor_Lf_Qh_1 = adjusted_mutual_info_score(gt_list, u_1_nor_Lf_Qh_individual_label_1)

print('average modularity_1 normalized L_F & Q_H score(K=12): ', modularity_1_nor_lf_qh)
print('average ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print('average purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print('average inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print('average NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)
print('average AMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', AMI_mbo_1_nor_Lf_Qh_1)

testarray_nor_Lf_Qh = ["modularity_1 normalized L_F & Q_H", "ARI_1 normalized L_F & Q_H",
             "purify for MMBO1 normalized L_F & Q_H", "inverse purify for MMBO1 normalized L_F & Q_H",
             "NMI for MMBO1 normalized L_F & Q_H ", "AMI for MMBO1 normalized L_F & Q_H"]


resultarray_nor_Lf_Qh = [modularity_1_nor_lf_qh, ARI_mbo_1_nor_Lf_Qh_1,
               purify_mbo_1_nor_Lf_Qh_1, inverse_purify_mbo_1_nor_Lf_Qh_1,
               NMI_mbo_1_nor_Lf_Qh_1, AMI_mbo_1_nor_Lf_Qh_1]

with open('MNIST_nor_LF_QH.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_nor_Lf_Qh)
    wr.writerow(resultarray_nor_Lf_Qh)


start_time_1_inner_nor_1 = time.time()

# MMBO1 with inner step & normalized L_F and gamma=1

u_inner_individual_1,num_repeat_inner = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, nor_graph_laplacian_1, nor_signless_laplacian_1,dt_inner, tol,target_size_1, inner_step_count)
print('u_inner number of iteration(K=12): ', num_repeat_inner)
u_inner_individual_label_1 = vector_to_labels(u_inner_individual_1)
#u_inner_individual_label_dict_1 = label_to_dict(u_inner_individual_label_1)
#u_inner_label_set = dict_to_list_set(u_inner_individual_label_dict_1)

print("MMBO1 with inner step & normalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_inner_nor_1))

#modularity_1_inner_individual_1 = co.modularity(u_inner_individual_label_dict_1,G)
#modularity_1_inner_1 = nx_comm.modularity(G,u_inner_label_set)
modularity_1_inner_1 = skn.clustering.modularity(W,u_inner_individual_label_1,resolution=0.5)
ARI_mbo_1_inner_1 = adjusted_rand_score(u_inner_individual_label_1, gt_list)
purify_mbo_1_inner_1 = purity_score(gt_list, u_inner_individual_label_1)
inverse_purify_mbo_1_inner_1 = inverse_purity_score(gt_list, u_inner_individual_label_1)
NMI_mbo_1_inner_1 = normalized_mutual_info_score(gt_list, u_inner_individual_label_1)
AMI_mbo_1_inner_1 = adjusted_mutual_info_score(gt_list, u_inner_individual_label_1)

print('average modularity_1 inner step score(K=12): ', modularity_1_inner_1)
print('average ARI_1 inner step score: ', ARI_mbo_1_inner_1)
print('average purify for MMBO1 inner step with \eta =1 : ', purify_mbo_1_inner_1)
print('average inverse purify for MMBO1 inner step with \eta =1 : ', inverse_purify_mbo_1_inner_1)
print('average NMI for MMBO1 inner step with \eta =1 : ', NMI_mbo_1_inner_1)
print('average AMI for MMBO1 inner step with \eta =1 : ', AMI_mbo_1_inner_1)

testarray_inner_1 = ["modularity_1 inner step", "ARI_1 inner step",
             "purify for MMBO1 inner step", "inverse purify for MMBO1 inner step",
             "NMI for MMBO1 inner step ", "AMI for MMBO1 inner step"]


resultarray_inner_1 = [modularity_1_inner_1, ARI_mbo_1_inner_1,
               purify_mbo_1_inner_1, inverse_purify_mbo_1_inner_1,
               NMI_mbo_1_inner_1, AMI_mbo_1_inner_1]

with open('MNIST_inner_1.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_inner_1)
    wr.writerow(resultarray_inner_1)


start_time_louvain = time.time()

# Louvain algorithm (can setting resolution gamma)
#louvain = skn.clustering.Louvain(resolution=1,modularity='newman')
#louvain_labels = louvain.fit_transform(W)

partition_Louvain = community_louvain.best_partition(G)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)

print("Louvain algorithm:-- %.3f seconds --" % (time.time() - start_time_louvain))


#modularity_louvain = nx_comm.modularity(G,louvain_label_set)
modularity_louvain = skn.clustering.modularity(W,louvain_array,resolution=0.5)
ARI_louvain = adjusted_rand_score(louvain_array, gt_list)
purify_louvain = purity_score(gt_list, louvain_array)
inverse_purify_louvain = inverse_purity_score(gt_list, louvain_array)
NMI_louvain = normalized_mutual_info_score(gt_list, louvain_array)
AMI_louvain = adjusted_mutual_info_score(gt_list, louvain_array)

print('average modularity Louvain score: ', modularity_louvain)
print('average ARI Louvain  score: ', ARI_louvain)
print('average purify for Louvain : ', purify_louvain)
print('average inverse purify for Louvain : ', inverse_purify_louvain)
print('average NMI for Louvain with \eta =1 : ', NMI_louvain)
print('average AMI for Louvain: ', AMI_louvain)

testarray_Louvain = ["modularity Louvain", "ARI Louvain",
             "purify for Louvain", "inverse purify for Louvain",
             "NMI for Louvain", "AMI for Louvain"]


resultarray_Louvain = [modularity_louvain, ARI_louvain,
               purify_louvain, inverse_purify_louvain,
               NMI_louvain, AMI_louvain]

with open('MNIST_Louvain.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_Louvain)
    wr.writerow(resultarray_Louvain)


start_time_CNM = time.time()

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
CNM_array = np.asarray(CNM_list)


print("CNM algorithm:-- %.3f seconds --" % (time.time() - start_time_CNM))


#modularity_CNM = nx_comm.modularity(G,partition_CNM_list)
modularity_CNM = skn.clustering.modularity(W,CNM_array,resolution=0.5)
ARI_CNM = adjusted_rand_score(CNM_list, gt_list)
purify_CNM = purity_score(gt_list, CNM_list)
inverse_purify_CNM = inverse_purity_score(gt_list, CNM_list)
NMI_CNM = normalized_mutual_info_score(gt_list, CNM_list)
AMI_CNM = adjusted_mutual_info_score(gt_list, CNM_list)

print('average modularity CNM score: ', modularity_CNM)
print('average ARI CNM  score: ', ARI_CNM)
print('average purify for CNM : ', purify_CNM)
print('average inverse purify for CNM : ', inverse_purify_CNM)
print('average NMI for CNM with \eta =1 : ', NMI_CNM)
print('average AMI for CNM: ', AMI_CNM)

testarray_CNM = ["modularity Louvain", "ARI Louvain",
             "purify for Louvain", "inverse purify for Louvain",
             "NMI for Louvain", "AMI for Louvain"]


resultarray_CNM = [modularity_CNM, ARI_CNM,
               purify_CNM, inverse_purify_CNM,
               NMI_CNM, AMI_CNM]

with open('MNIST_CNM.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_CNM)
    wr.writerow(resultarray_CNM)


start_time_spectral_clustering = time.time()

# Spectral clustering with k-means
sc = SpectralClustering(n_clusters=11, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)

#ass_dict = label_to_dict (assignment)
#sc_label_set = dict_to_list_set(ass_dict)

print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

#modularity_spectral_clustering = nx_comm.modularity(G,sc_label_set)
modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=0.5)
ARI_spectral_clustering = adjusted_rand_score(assignment, gt_list)
purify_spectral_clustering = purity_score(gt_list, assignment)
inverse_purify_spectral_clustering = inverse_purity_score(gt_list, assignment)
NMI_spectral_clustering = normalized_mutual_info_score(gt_list, assignment)
AMI_spectral_clustering = adjusted_mutual_info_score(gt_list, assignment)


print('average modularity Spectral clustering score(K=12): ', modularity_spectral_clustering)
print('average ARI Spectral clustering  score: ', ARI_spectral_clustering)
print('average purify for Spectral clustering : ', purify_spectral_clustering)
print('average inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
print('average NMI for Spectral clustering with \eta =1 : ', NMI_spectral_clustering)
print('average AMI for Spectral clustering: ', AMI_spectral_clustering)

testarray_sc = ["modularity Spectral clustering", "ARI Spectral clustering",
             "purify for Spectral clustering", "inverse purify for Spectral clustering",
             "NMI for Spectral clustering", "AMI for Spectral clustering"]


resultarray_sc = [modularity_spectral_clustering, ARI_spectral_clustering,
               purify_spectral_clustering, inverse_purify_spectral_clustering,
               NMI_spectral_clustering, AMI_spectral_clustering]

with open('MNIST_spectral_clustering.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_sc)
    wr.writerow(resultarray_sc)