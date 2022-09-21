from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import random
from sklearn.decomposition import PCA
import sknetwork as skn
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import time
from community import community_louvain
from Nystrom_extension_QR import nystrom_QR_l_sym, nystrom_QR_l_mix_sym_rw, nystrom_QR_l_mix_B_sym_rw
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
import graphlearning as gl
import cdlib
from cdlib.algorithms import louvain
from cdlib import evaluation, NodeClustering
from MMBO_and_HU import MMBO_using_projection, MMBO_using_finite_differendce,HU_mmbo_method
from utils import vector_to_labels, labels_to_vector, label_to_dict, dict_to_list_set, purity_score, inverse_purity_score, generate_initial_value_multiclass
from Nystrom_QR_test import nystrom_QR_l_mix_sym_rw_ER_null, nystrom_QR_l_mix_B_sym_rw_ER_null



# Example 2: MNIST (with Erdős–Rényi model)
print('Using the Erdős_Rényi model as null model')

# Parameter setting

# num_communities is found by Louvain
# choose m = num_communities
tol = 1e-5
modularity_tol = 1e-4
N_t = 5
gamma = 0.5
tau = 0.02
num_nodes = 70000


# Load MNIST data, ground truth, and build 10-nearest neighbor weight matrix
data, gt_labels = gl.datasets.load('mnist')
#gt_vec = labels_to_vector(gt_labels)

gt_labels_list = list(gt_labels)

# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_labels_list)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_labels_list))


pca = PCA(n_components = 50)
Z_training = pca.fit_transform(data)
W = gl.weightmatrix.knn(Z_training, 10, symmetrize=True)
G = nx.convert_matrix.from_scipy_sparse_matrix(W)


# First run the Louvain method in order to get the number of clusters
sum_louvain_cluster =0
sum_time_louvain=0
sum_modularity_louvain =0
sum_ER_modularity_louvain =0
sum_ARI_louvain = 0
sum_purity_louvain = 0
sum_inverse_purity_louvain = 0
sum_NMI_louvain = 0



for _ in range(1):
    start_time_louvain = time.time()
    partition_Louvain = community_louvain.best_partition(G, resolution=gamma)    # returns a dict
    time_louvain = time.time() - start_time_louvain
    #print("Louvain:-- %.3f seconds --" % (time_louvain))

    louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
    louvain_array = np.asarray(louvain_list)
    louvain_cluster = len(np.unique(louvain_array))

    louvain_partition_list = dict_to_list_set(partition_Louvain)
    louvain_communities = NodeClustering(louvain_partition_list, graph=None)


    ER_modularity_louvain = evaluation.erdos_renyi_modularity(G,louvain_communities)[2]
    #modularity_louvain = evaluation.newman_girvan_modularity(G,louvain_communities)[2]
    #modularity_louvain = skn.clustering.modularity(W,louvain_array,resolution=gamma)
    ARI_louvain = adjusted_rand_score(louvain_array, gt_labels)
    purify_louvain = purity_score(gt_labels, louvain_array)
    inverse_purify_louvain = inverse_purity_score(gt_labels, louvain_array)
    NMI_louvain = normalized_mutual_info_score(gt_labels, louvain_array)

    sum_louvain_cluster += louvain_cluster
    sum_time_louvain += time_louvain
    #sum_modularity_louvain += modularity_louvain
    sum_ER_modularity_louvain += ER_modularity_louvain
    sum_ARI_louvain += ARI_louvain
    sum_purity_louvain += purify_louvain
    sum_inverse_purity_louvain += inverse_purify_louvain
    sum_NMI_louvain += NMI_louvain

average_louvain_cluster = sum_louvain_cluster / 1
average_time_louvain = sum_time_louvain / 1
#average_modularity_louvain = sum_modularity_louvain / 1
average_ER_modularity_louvain = sum_ER_modularity_louvain / 1
average_ARI_louvain = sum_ARI_louvain / 1
average_purify_louvain = sum_purity_louvain / 1
average_inverse_purify_louvain = sum_inverse_purity_louvain / 1
average_NMI_louvain = sum_NMI_louvain / 1


print('average_time_louvain: ', average_time_louvain)
#print('average_modularity_louvain: ', average_modularity_louvain)
print('average_ER_modularity_louvain: ', average_ER_modularity_louvain)
print('average_ARI_louvain: ', average_ARI_louvain)
print('average_purify_louvain: ', average_purify_louvain)
print('average_inverse_purify_louvain: ', average_inverse_purify_louvain)
print('average_NMI_louvain: ', average_NMI_louvain)

num_communities  = round(average_louvain_cluster)
m = num_communities


# CNM algorithm (can setting resolution gamma)
sum_time_CNM =0
sum_modularity_CNM =0
sum_ER_modularity_CNM =0
sum_ARI_CNM = 0
sum_purity_CNM = 0
sum_inverse_purity_CNM = 0
sum_NMI_CNM = 0

for _ in range(1):
    start_time_CNM = time.time()
    partition_CNM = nx_comm.greedy_modularity_communities(G, resolution=gamma)
    time_CNM = time.time() - start_time_CNM
    #print("CNM algorithm:-- %.3f seconds --" % (time.time() - start_time_CNM))

    partition_CNM_list = [list(x) for x in partition_CNM]
    partition_CNM_expand = sum(partition_CNM_list, [])

    num_cluster_CNM = []
    for cluster in range(len(partition_CNM_list)):
        for number_CNM in range(len(partition_CNM_list[cluster])):
            num_cluster_CNM.append(cluster)

    CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))

    partition_CNM_sort = np.sort(partition_CNM_expand)
    CNM_list_sorted = []
    for CNM_element in partition_CNM_sort:
        CNM_list_sorted.append(CNM_dict[CNM_element])
    CNM_array_sorted = np.asarray(CNM_list_sorted)
    
    CNM_vec = labels_to_vector(CNM_array_sorted)
    CNM_coms = NodeClustering(partition_CNM_list, graph=None)
    
    ER_modularity_CNM = evaluation.erdos_renyi_modularity(G, CNM_coms)[2]
    #modularity_CNM = evaluation.newman_girvan_modularity(G, CNM_coms)[2]
    #modularity_CNM = skn.clustering.modularity(W,CNM_array_sorted,resolution=gamma)
    ARI_CNM = adjusted_rand_score(CNM_array_sorted, gt_labels)
    purify_CNM = purity_score(gt_labels, CNM_array_sorted)
    inverse_purify_CNM = inverse_purity_score(gt_labels, CNM_array_sorted)
    NMI_CNM = normalized_mutual_info_score(gt_labels, CNM_array_sorted)

    sum_time_CNM += time_CNM
    #sum_modularity_CNM += modularity_CNM
    sum_ER_modularity_CNM += ER_modularity_CNM
    sum_ARI_CNM += ARI_CNM
    sum_purity_CNM += purify_CNM
    sum_inverse_purity_CNM += inverse_purify_CNM
    sum_NMI_CNM += NMI_CNM


average_time_CNM = sum_time_CNM / 1
#average_modularity_CNM = sum_modularity_CNM / 1
average_ER_modularity_CNM = sum_ER_modularity_CNM / 1
average_ARI_CNM = sum_ARI_CNM / 1
average_purity_CNM = sum_purity_CNM / 1
average_inverse_purity_CNM = sum_inverse_purity_CNM / 1
average_NMI_CNM = sum_NMI_CNM / 1


print('CNM')
print('average_time_CNM: ', average_time_CNM)
#print('average_modularity_CNM: ', average_modularity_CNM)
print('average_ER_modularity_CNM: ', average_ER_modularity_CNM)
print('average_ARI_CNM: ', average_ARI_CNM)
print('average_purity_CNM: ', average_purity_CNM)
print('average_inverse_purity_CNM: ', average_inverse_purity_CNM)
print('average_NMI_CNM: ', average_NMI_CNM)


# Spectral clustering with k-means
sum_time_sc=0
sum_modularity_sc =0
sum_ER_modularity_sc =0
sum_ARI_spectral_clustering = 0
sum_purify_spectral_clustering = 0
sum_inverse_purify_spectral_clustering = 0
sum_NMI_spectral_clustering = 0

for _ in range(1):
    start_time_spectral_clustering = time.time()
    sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed')
    assignment = sc.fit_predict(W)
    time_sc = time.time() - start_time_spectral_clustering
    #print("spectral clustering algorithm:-- %.3f seconds --" % (time_sc))

    ass_vec = labels_to_vector(assignment)
    ass_dict = label_to_dict(assignment)
    ass_list = dict_to_list_set(ass_dict)
    ass_coms = NodeClustering(ass_list, graph=None)
    
    ER_modularity_spectral_clustering = evaluation.erdos_renyi_modularity(G, ass_coms)[2]
    #modularity_spectral_clustering = evaluation.newman_girvan_modularity(G, ass_coms)[2]
    #modularity_spectral_clustering = skn.clustering.modularity(W,assignment,resolution=1)
    ARI_spectral_clustering = adjusted_rand_score(assignment, gt_labels)
    purify_spectral_clustering = purity_score(gt_labels, assignment)
    inverse_purify_spectral_clustering = inverse_purity_score(gt_labels, assignment)
    NMI_spectral_clustering = normalized_mutual_info_score(gt_labels, assignment)

print('Spectral clustering')
print('time_sc: ', time_sc)
print('ER_modularity_sc: ', ER_modularity_spectral_clustering)
#print('modularity Spectral clustering score: ', modularity_spectral_clustering)
print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
print('purify for Spectral clustering : ', purify_spectral_clustering)
print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
print('NMI for Spectral clustering: ', NMI_spectral_clustering)


