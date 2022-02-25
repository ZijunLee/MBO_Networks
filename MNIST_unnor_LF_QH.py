import numpy as np
import graphlearning as gl
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
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

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1, rw_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(W,num_communities,m,eta_1,target_size=None)

# MMBO 1 with unnormalized L_F and gamma=1
mbo_accumulator_1_unnor_individual = 0
ARI_1_unnor_lf_accumulator =0
purify_1_unnor_lf_accumulator_1 =0
inverse_purify_1_unnor_lf_accumulator_1 =0
NMI_1_unnor_lf_accumulator_1 =0
AMI_1_unnor_lf_accumulator_1 =0

#for _ in range(30):
u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)   
u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)

modularity_1_unnor_individual = co.modularity(u_1_unnor_individual_label_dict,G)
ARI_mbo_1_unnor_lf = adjusted_rand_score(u_1_unnor_individual_label, gt_list)
purify_mbo_1_unnor_lf_1 = purity_score(gt_list, u_1_unnor_individual_label)
inverse_purify_mbo_1_unnor_lf_1 = inverse_purity_score(gt_list, u_1_unnor_individual_label)
NMI_mbo_1_unnor_lf_1 = normalized_mutual_info_score(gt_list, u_1_unnor_individual_label)
AMI_mbo_1_unnor_lf_1 = adjusted_mutual_info_score(gt_list, u_1_unnor_individual_label)

    #mbo_accumulator_1_unnor_individual += modularity_1_unnor_individual
    #ARI_1_unnor_lf_accumulator += ARI_mbo_1_unnor_lf
    #purify_1_unnor_lf_accumulator_1 += purify_mbo_1_unnor_lf_1
    #inverse_purify_1_unnor_lf_accumulator_1 += inverse_purify_mbo_1_unnor_lf_1
    #NMI_1_unnor_lf_accumulator_1 += NMI_mbo_1_unnor_lf_1
    #AMI_1_unnor_lf_accumulator_1 += AMI_mbo_1_unnor_lf_1

#average_mbo_1_unnor = mbo_accumulator_1_unnor_individual / 30
#average_ARI_1_unnor = ARI_1_unnor_lf_accumulator / 30
#average_purify_1_unnor_1 = purify_1_unnor_lf_accumulator_1 / 30
#average_inverse_purify_1_unnor_1 = inverse_purify_1_unnor_lf_accumulator_1 / 30
#average_NMI_1_unnor_1 = NMI_1_unnor_lf_accumulator_1 / 30
#average_AMI_1_unnor_1 = AMI_1_unnor_lf_accumulator_1 / 30

print('average modularity_1 unnormalized L_F & Q_H score: ', modularity_1_unnor_individual)
print('average ARI_1 unnormalized L_F & Q_H score: ', ARI_mbo_1_unnor_lf)
print('average purify for MMBO1 unnormalized L_F with \eta =1 : ', purify_mbo_1_unnor_lf_1)
print('average inverse purify for MMBO1 unnormalized L_F with \eta =1 : ', inverse_purify_mbo_1_unnor_lf_1)
print('average NMI for MMBO1 unnormalized L_F with \eta =1 : ', NMI_mbo_1_unnor_lf_1)
print('average AMI for MMBO1 unnormalized L_F with \eta =1 : ', AMI_mbo_1_unnor_lf_1)

testarray = ["average modularity_1 unnormalized L_F & Q_H score", "average ARI_1 unnormalized L_F & Q_H score",
             "average purify for MMBO1 unnormalized L_F", "average inverse purify for MMBO1 unnormalized L_F",
             "average NMI for MMBO1 unnormalized L_F", "average AMI for MMBO1 unnormalized L_F"]

#resultarray = [average_mbo_1_unnor, average_ARI_1_unnor,
#               average_purify_1_unnor_1, average_inverse_purify_1_unnor_1,
#               average_NMI_1_unnor_1, average_AMI_1_unnor_1]

resultarray = [modularity_1_unnor_individual, ARI_mbo_1_unnor_lf,
               purify_mbo_1_unnor_lf_1, inverse_purify_mbo_1_unnor_lf_1,
               NMI_mbo_1_unnor_lf_1, AMI_mbo_1_unnor_lf_1]

with open('MNIST_unnor_LF_QH.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray)
    wr.writerow(resultarray)