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

#Load labels, knndata, and build 10-nearest neighbor weight matrix
W = gl.weightmatrix.knn('mnist', 10, metric='vae')
W_dense = W.todense()
print(W_dense.shape)
print(type(W_dense))

gt_labels = gl.datasets.load('mnist', labels_only=True)
gt_list = gt_labels.tolist()  
#print('gt shape: ', type(gt_list))

# convert a list to a dict
gt_label_dict = []
len_gt_label = []

for e in range(len(gt_list)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_list))     # gt_label_dict is a dict


G = nx.convert_matrix.from_numpy_matrix(W_dense)
print(type(G))


# Louvain algorithm (can setting resolution gamma)
partition_Louvain = community_louvain.best_partition(G, resolution=1)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_label_set = dict_to_list_set(partition_Louvain)

print('louvain list set: ', louvain_label_set)