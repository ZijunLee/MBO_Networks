import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_2, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
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


G = nx.convert_matrix.from_scipy_sparse_matrix(W)
print(type(G))

adj_mat = nx.convert_matrix.to_numpy_matrix(G)


## parameter setting
dt_inner = 0.1
num_communities = 11
m = 5 * num_communities
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


start_time_2_nor_1 = time.time()

# mmbo 2 with normalized & gamma = 1

u_2_individual_1, num_repeat_2_1 = mbo_modularity_2(num_communities, m, adj_mat, tol,eta_1,eps=1) 
u_2_individual_label_1 = vector_to_labels(u_2_individual_1)
u_2_individual_label_dict_1 = label_to_dict(u_2_individual_label_1)
#u_2_nor_label_set = dict_to_list_set(u_2_individual_label_dict_1)

print("mmbo 2 with normalized & gamma = 1:-- %.3f seconds --" % (time.time() - start_time_2_nor_1))

#modularity_2_nor_individual_1 = co.modularity(u_2_individual_label_dict_1,G)
#modularity_2_nor_lf_qh = nx_comm.modularity(G,u_2_nor_label_set)
modularity_2_nor_lf_qh = skn.clustering.modularity(W,u_2_individual_label_1,resolution=0.5)
ARI_mbo_2_nor_1 = adjusted_rand_score(u_2_individual_label_1, gt_list)
purify_mbo_2_nor_1 = purity_score(gt_list, u_2_individual_label_1)
inverse_purify_mbo_2_nor_1 = inverse_purity_score(gt_list, u_2_individual_label_1)
NMI_mbo_2_nor_1 = normalized_mutual_info_score(gt_list, u_2_individual_label_1)
AMI_mbo_2_nor_1 = adjusted_mutual_info_score(gt_list, u_2_individual_label_1)


print('average mmbo 2 with normalized score: ', modularity_2_nor_lf_qh)
print('average ARI for mmbo 2 with normalized score: ', ARI_mbo_2_nor_1)
print('average purify for mmbo 2 with normalized with \eta =1 : ', purify_mbo_2_nor_1)
print('average inverse purify for mmbo 2 with normalized with \eta =1 : ', inverse_purify_mbo_2_nor_1)
print('average NMI for mmbo 2 with normalized with \eta =1 : ', NMI_mbo_2_nor_1)
print('average AMI for mmbo 2 with normalized with \eta =1 : ', AMI_mbo_2_nor_1)



testarray_2_nor_Lf_Qh = ["average mmbo 2 with normalized score", "average ARI_1 mmbo 2 with normalized score",
             "average purify for mmbo 2 with normalized", "average inverse purify for mmbo 2 with normalized",
             "average NMI for mmbo 2 with normalized", "average AMI for mmbo 2 with normalized"]

#resultarray = [average_mbo_1_unnor, average_ARI_1_unnor,
#               average_purify_1_unnor_1, average_inverse_purify_1_unnor_1,
#               average_NMI_1_unnor_1, average_AMI_1_unnor_1]

resultarray_2_nor_Lf_Qh = [modularity_2_nor_lf_qh, ARI_mbo_2_nor_1,
               purify_mbo_2_nor_1, inverse_purify_mbo_2_nor_1,
               NMI_mbo_2_nor_1, AMI_mbo_2_nor_1]

with open('MNIST_MMBO2_nor_LF_QH_10.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_2_nor_Lf_Qh)
    wr.writerow(resultarray_2_nor_Lf_Qh)
