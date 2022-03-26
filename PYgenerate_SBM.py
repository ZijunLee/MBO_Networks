from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from MBO_Network import mbo_modularity_1, mbo_modularity_2,data_generator,SSBM_own,adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
from MBO_Network import mbo_modularity_hu_original,MMBO2_preliminary
from graph_mbo.utils import spectral_clustering,vector_to_labels, get_modularity_original,get_modularity,labels_to_vector,label_to_dict,purity_score,inverse_purity_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
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
import sknetwork as skn
from community import community_louvain


# parameter setting
dt_inner = 1
num_communities = 10
m = 1 * num_communities
m_100 = 100
tol = 0.0003
eta_1 = 1
eta_06 = 0.6
eta_05 = 0.5
eta_03 = 1.3
inner_step_count =3

N = 3000
num_nodes_each_cluster = int(N/num_communities)
#print('num_nodes_each_cluster type: ', type(num_nodes_each_cluster))


sizes = []

for i in range(10):
    sizes.append(num_nodes_each_cluster)

#print(len(sizes))

all_one_matrix = np.ones((len(sizes),len(sizes)))
diag_matrix = np.diag(np.full(len(sizes),1))
probs = 0.1 * all_one_matrix + 0.2 * diag_matrix
print(probs)

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
adj_mat_nparray = nx.convert_matrix.to_numpy_array(G)
#print('adj type: ', type(adj_mat))






num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1,random_walk_nor_lap_1, signless_laplacian_null_model_1, nor_signless_laplacian_1, rw_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat_nparray,num_communities,m,eta_1,target_size=None)
#num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, random_walk_nor_lap_06, signless_laplacian_null_model_06, nor_signless_laplacian_06,rw_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_06,target_size=None)
#num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, random_walk_nor_lap_05, signless_laplacian_null_model_05, nor_signless_laplacian_05, rw_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_05,target_size=None)

num_nodes_B_1, m_B_1, target_size_B_1, graph_laplacian_positive_B_1, sym_lap_positive_B_1, rw_nor_lap_positive_B_1, signless_lap_neg_B_1, sym_signless_lap_negative_B_1, rw_signless_lap_negative_B_1 = MMBO2_preliminary(adj_mat, num_communities,m,eta_1)


start_time_1_unnor_1 = time.time()

# mmbo 1 with unnormalized L_F and gamma=1
#u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
#                                                tol, target_size_1,eta_1)
#print("mmbo 1 with unnormalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_unnor_1))
#u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
#u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)


#modularity_1_unnor_individual = skn.clustering.modularity(adj_mat_nparray,u_1_unnor_individual_label,resolution=1)
#ARI_mbo_1_unnor_lf = adjusted_rand_score(u_1_unnor_individual_label, gt_membership)
#purify_mbo_1_unnor_lf_1 = purity_score(gt_membership, u_1_unnor_individual_label)
#inverse_purify_mbo_1_unnor_lf_1 = inverse_purity_score(gt_membership, u_1_unnor_individual_label)
#NMI_mbo_1_unnor_lf_1 = normalized_mutual_info_score(gt_membership, u_1_unnor_individual_label)
#AMI_mbo_1_unnor_lf_1 = adjusted_mutual_info_score(gt_membership, u_1_unnor_individual_label)

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

#print(' modularity_1 unnormalized L_F & Q_H score: ', modularity_1_unnor_individual)
#print(' ARI_1 unnormalized L_F & Q_H score: ', ARI_mbo_1_unnor_lf)
#print(' purify for MMBO1 unnormalized L_F with \eta =1 : ', purify_mbo_1_unnor_lf_1)
#print(' inverse purify for MMBO1 unnormalized L_F with \eta =1 : ', inverse_purify_mbo_1_unnor_lf_1)
#print(' NMI for MMBO1 unnormalized L_F with \eta =1 : ', NMI_mbo_1_unnor_lf_1)
#print(' AMI for MMBO1 unnormalized L_F with \eta =1 : ', AMI_mbo_1_unnor_lf_1)



start_time_1_nor_Lf_Qh_1 = time.time()

# MMBO1 with normalized L_F & Q_H and gamma=1
#mbo_accumulator_1_nor_Lf_Qh_individual_1 =0
#ARI_1_nor_Lf_Qh_accumulator_1 =0
#purify_1_nor_Lf_Qh_accumulator_1 =0
#inverse_purify_1_nor_Lf_Qh_accumulator_1 =0
#NMI_1_nor_Lf_Qh_accumulator_1 =0
#AMI_1_nor_Lf_Qh_accumulator_1 =0

#for _ in range(30):
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1,dt_inner, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                tol, target_size_1,eta_1)

print("MMBO1 with normalized L_F & Q_H and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_nor_Lf_Qh_1))
print('number of ieration of MMBO1 with sym L_F & Q_H: ',num_repeat_1_nor_Lf_Qh_1)   
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_individual_label_1)


modularity_1_nor_Lf_Qh_individual_1 = skn.clustering.modularity(adj_mat_nparray,u_1_nor_Lf_Qh_individual_label_1,resolution=1)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_membership)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
AMI_mbo_1_nor_Lf_Qh_1 = adjusted_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)

    #mbo_accumulator_1_nor_Lf_Qh_individual_1 += modularity_1_nor_Lf_Qh_individual_1
    #ARI_1_nor_Lf_Qh_accumulator_1 += ARI_mbo_1_nor_Lf_Qh_1
    #purify_1_nor_Lf_Qh_accumulator_1 += purify_mbo_1_nor_Lf_Qh_1
    #inverse_purify_1_nor_Lf_Qh_accumulator_1 += inverse_purify_mbo_1_nor_Lf_Qh_1
    #NMI_1_nor_Lf_Qh_accumulator_1 += NMI_mbo_1_nor_Lf_Qh_1
    #AMI_1_nor_Lf_Qh_accumulator_1 += AMI_mbo_1_nor_Lf_Qh_1

#average_mbo_1_nor_lf_Qh_1 = mbo_accumulator_1_nor_Lf_Qh_individual_1 / 30
#average_ARI_1_nor_lf_Qh_1 = ARI_1_nor_Lf_Qh_accumulator_1 / 30
#average_purify_1_nor_lf_Qh_1 = purify_1_nor_Lf_Qh_accumulator_1 / 30
#average_inverse_purify_1_nor_lf_Qh_1 = inverse_purify_1_nor_Lf_Qh_accumulator_1 / 30
#average_NMI_1_nor_lf_Qh_1 = NMI_1_nor_Lf_Qh_accumulator_1 / 30
#average_AMI_1_nor_lf_Qh_1 = AMI_1_nor_Lf_Qh_accumulator_1 / 30

print(' modularity_1 normalized L_F & Q_H score: ', modularity_1_nor_Lf_Qh_individual_1)
print(' ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print(' purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print(' inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print(' NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)
print(' AMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', AMI_mbo_1_nor_Lf_Qh_1)



start_time_1_rw_Lf_1 = time.time()

# MMBO1 with random walk L_F and gamma=1

#u_1_rw_individual_1,num_repeat_1_rw = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1,dt_inner, random_walk_nor_lap_1,signless_laplacian_null_model_1, 
#                                                tol, target_size_1,eta_1, eps=1)     
#u_1_rw_individual_label_1 = vector_to_labels(u_1_rw_individual_1)
#print('u_1_rw_individual_label_1: ', u_1_rw_individual_label_1)
#u_1_rw_individual_label_dict_1 = label_to_dict(u_1_rw_individual_label_1)

#print("MMBO1 with random walk L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_rw_Lf_1))

#modularity_1_rw_individual_1 = co.modularity(u_1_rw_individual_label_dict_1,G)
#ARI_mbo_1_rw_1 = adjusted_rand_score(u_1_rw_individual_label_1, gt_membership)
#purify_mbo_1_rw_1 = purity_score(gt_membership, u_1_rw_individual_label_1)
#inverse_purify_mbo_1_rw_1 = inverse_purity_score(gt_membership, u_1_rw_individual_label_1)
#NMI_mbo_1_rw_1 = normalized_mutual_info_score(gt_membership, u_1_rw_individual_label_1)
#AMI_mbo_1_rw_1 = adjusted_mutual_info_score(gt_membership, u_1_rw_individual_label_1)

#mbo_accumulator_1_rw_individual_1 += modularity_1_rw_individual_1
#ARI_1_rw_accumulator_1 += ARI_mbo_1_rw_1
#purify_1_rw_accumulator_1 += purify_mbo_1_rw_1
#inverse_purify_1_rw_accumulator_1 += inverse_purify_mbo_1_rw_1
#NMI_1_rw_accumulator_1 += NMI_mbo_1_rw_1
#AMI_1_rw_accumulator_1 += AMI_mbo_1_rw_1

#average_mbo_1_rw_1 = mbo_accumulator_1_rw_individual_1 / 30
#average_ARI_1_rw_1 = ARI_1_rw_accumulator_1 / 30
#average_purify_1_rw_1 = purify_1_rw_accumulator_1 / 30
#average_inverse_purify_1_rw_1 = inverse_purify_1_rw_accumulator_1 / 30
#average_NMI_1_rw_1 = NMI_1_rw_accumulator_1 / 30
#average_AMI_1_rw_1 = AMI_1_rw_accumulator_1 / 30

#print(' modularity_1 random walk L_F score: ', modularity_1_rw_individual_1)
#print(' ARI_1 random walk L_F score: ', ARI_mbo_1_rw_1)
#print(' purify for MMBO1 random walk L_F with \eta =1 : ', purify_mbo_1_rw_1)
#print(' inverse purify for MMBO1 random walk L_F with \eta =1 : ', inverse_purify_mbo_1_rw_1)
#print(' NMI for MMBO1 random walk L_F with \eta =1 : ', NMI_mbo_1_rw_1)
#print(' AMI for MMBO1 random walk L_F with \eta =1 : ', AMI_mbo_1_rw_1)



start_time_1_rw_Lf_Qh_1 = time.time()

# MMBO1 with random walk L_F & Q_H and gamma=1

#u_1_rw_Lf_Qh_individual_1,num_repeat_1_rw_Lf_Qh = mbo_modularity_1(num_nodes_1,num_communities, m_1,degree_1, dt_inner,random_walk_nor_lap_1,rw_signless_laplacian_1, 
#                                                tol, target_size_1,eta_1, eps=1)     
#u_1_rw_Lf_Qh_individual_label_1 = vector_to_labels(u_1_rw_Lf_Qh_individual_1)
#u_1_rw_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_rw_Lf_Qh_individual_label_1)

#print("MMBO1 with random walk L_F & Q_H and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_rw_Lf_Qh_1))

#modularity_1_rw_Lf_Qh_individual_1 = skn.clustering.modularity(adj_mat_nparray,u_1_rw_Lf_Qh_individual_label_1,resolution=1)
#ARI_mbo_1_rw_Lf_Qh_1 = adjusted_rand_score(u_1_rw_Lf_Qh_individual_label_1, gt_membership)
#purify_mbo_1_rw_Lf_Qh_1 = purity_score(gt_membership, u_1_rw_Lf_Qh_individual_label_1)
#inverse_purify_mbo_1_rw_Lf_Qh_1 = inverse_purity_score(gt_membership, u_1_rw_Lf_Qh_individual_label_1)
#NMI_mbo_1_rw_Lf_Qh_1 = normalized_mutual_info_score(gt_membership, u_1_rw_Lf_Qh_individual_label_1)
#AMI_mbo_1_rw_Lf_Qh_1 = adjusted_mutual_info_score(gt_membership, u_1_rw_Lf_Qh_individual_label_1)


#print('average modularity_1 random walk L_F & Q_H score: ', modularity_1_rw_Lf_Qh_individual_1)
#print('average ARI_1 random walk L_F & Q_H score: ', ARI_mbo_1_rw_Lf_Qh_1)
#print('average purify for MMBO1 random walk L_F & Q_H with \eta =1 : ', purify_mbo_1_rw_Lf_Qh_1)
#print('average inverse purify for MMBO1 random walk L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_rw_Lf_Qh_1)
#print('average NMI for MMBO1 random walk L_F & Q_H with \eta =1 : ', NMI_mbo_1_rw_Lf_Qh_1)
#print('average AMI for MMBO1 random walk L_F & Q_H with \eta =1 : ', AMI_mbo_1_rw_Lf_Qh_1)


#start_time_2_unnor_1 = time.time()

# mmbo2 with unnormalized L_F & Q_H
#u_2_unnor_1, num_repeat_2_unnor_1 = mbo_modularity_1(num_nodes_B_1, num_communities, m_B_1,degree_1,dt_inner,  
#                                            graph_laplacian_positive_B_1, signless_lap_neg_B_1,tol,target_size_B_1,eta_1)
 
#u_2_unnor_label_1 = vector_to_labels(u_2_unnor_1)
#u_2_unnor_label_dict_1 = label_to_dict(u_2_unnor_label_1)

#print("mmbo 2 with unnormalized & gamma = 1:-- %.3f seconds --" % (time.time() - start_time_2_unnor_1))

#modularity_2_unnor_1 = skn.clustering.modularity(adj_mat_nparray,u_2_unnor_label_1,resolution=1)
#ARI_mbo_2_unnor_1 = adjusted_rand_score(u_2_unnor_label_1, gt_membership)
#purify_mbo_2_unnor_1 = purity_score(gt_membership, u_2_unnor_label_1)
#inverse_purify_mbo_2_unnor_1 = inverse_purity_score(gt_membership, u_2_unnor_label_1)
#NMI_mbo_2_unnor_1 = normalized_mutual_info_score(gt_membership, u_2_unnor_label_1)
#AMI_mbo_2_unnor_1 = adjusted_mutual_info_score(gt_membership, u_2_unnor_label_1)

#print(' mmbo 2 with unnormalized score: ', modularity_2_unnor_1)
#print(' ARI for mmbo 2 with unnormalized score: ', ARI_mbo_2_unnor_1)
#print(' purify for mmbo 2 with unnormalized with \eta =1 : ', purify_mbo_2_unnor_1)
#print(' inverse purify for mmbo 2 with unnormalized with \eta =1 : ', inverse_purify_mbo_2_unnor_1)
#print(' NMI for mmbo 2 with unnormalized with \eta =1 : ', NMI_mbo_2_unnor_1)
#print(' AMI for mmbo 2 with unnormalized with \eta =1 : ', AMI_mbo_2_unnor_1)


#start_time_2_nor_1 = time.time()

# mmbo 2 with normalized & gamma = 1

#u_2_sym_1, num_repeat_2_sym_1 = mbo_modularity_1(num_nodes_1, num_communities, m_1,degree_1,dt_inner,  
#                                            sym_lap_positive_B_1, sym_signless_lap_negative_B_1,tol,target_size_1,eta_1)
#u_2_sym_label_1 = vector_to_labels(u_2_sym_1)
#u_2_sym_label_dict_1 = label_to_dict(u_2_sym_label_1)

#print("mmbo 2 with sym normalized & gamma = 1:-- %.3f seconds --" % (time.time() - start_time_2_nor_1))

#modularity_2_nor_individual_1 = skn.clustering.modularity(adj_mat_nparray,u_2_sym_label_1,resolution=1)
#ARI_mbo_2_nor_1 = adjusted_rand_score(u_2_sym_label_1, gt_membership)
#purify_mbo_2_nor_1 = purity_score(gt_membership, u_2_sym_label_1)
#inverse_purify_mbo_2_nor_1 = inverse_purity_score(gt_membership, u_2_sym_label_1)
#NMI_mbo_2_nor_1 = normalized_mutual_info_score(gt_membership, u_2_sym_label_1)
#AMI_mbo_2_nor_1 = adjusted_mutual_info_score(gt_membership, u_2_sym_label_1)

#print(' mmbo 2 with normalized score: ', modularity_2_nor_individual_1)
#print(' ARI for mmbo 2 with normalized score: ', ARI_mbo_2_nor_1)
#print(' purify for mmbo 2 with normalized with \eta =1 : ', purify_mbo_2_nor_1)
#print(' inverse purify for mmbo 2 with normalized with \eta =1 : ', inverse_purify_mbo_2_nor_1)
#print(' NMI for mmbo 2 with normalized with \eta =1 : ', NMI_mbo_2_nor_1)
#print(' AMI for mmbo 2 with normalized with \eta =1 : ', AMI_mbo_2_nor_1)



start_time_2_rw_nor_1 = time.time()

# mmbo2 with random walk L_F & Q_H
#u_2_rw_1, num_repeat_2_rw_1 = mbo_modularity_1(num_nodes_B_1, num_communities, m_B_1,degree_1,dt_inner,
#                                            rw_nor_lap_positive_B_1, rw_signless_lap_negative_B_1,tol,target_size_B_1,eta_1)
#u_2_rw_label_1 = vector_to_labels(u_2_rw_1)
#u_2_rw_label_dict_1 = label_to_dict(u_2_rw_label_1)

#print("mmbo 2 with random walk & gamma = 1:-- %.3f seconds --" % (time.time() - start_time_2_rw_nor_1))

#modularity_2_rw_1 = skn.clustering.modularity(adj_mat_nparray,u_2_rw_label_1,resolution=1)
#ARI_mbo_2_rw_1 = adjusted_rand_score(u_2_rw_label_1, gt_membership)
#purify_mbo_2_rw_1 = purity_score(gt_membership, u_2_rw_label_1)
#inverse_purify_mbo_2_rw_1 = inverse_purity_score(gt_membership, u_2_rw_label_1)
#NMI_mbo_2_rw_1 = normalized_mutual_info_score(gt_membership, u_2_rw_label_1)
#AMI_mbo_2_rw_1 = adjusted_mutual_info_score(gt_membership, u_2_rw_label_1)


#print(' mmbo 2 with random walk score: ', modularity_2_rw_1)
#print(' ARI for mmbo 2 with random walk score: ', ARI_mbo_2_rw_1)
#print(' purify for mmbo 2 with random walk with \eta =1 : ', purify_mbo_2_rw_1)
#print(' inverse purify for mmbo 2 with random walk with \eta =1 : ', inverse_purify_mbo_2_rw_1)
#print(' NMI for mmbo 2 with random walk with \eta =1 : ', NMI_mbo_2_rw_1)
#print(' AMI for mmbo 2 with random walk with \eta =1 : ', AMI_mbo_2_rw_1)



start_time_1_inner_unnor_1 = time.time()

# MMBO1 with inner step & unnormalized L_F and gamma=1

#u_inner_unnor_1,num_repeat_inner_unnor = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, graph_laplacian_1, signless_laplacian_null_model_1,
#                                                          dt_inner, tol,target_size_1, inner_step_count)
#u_inner_unnor_label_1 = vector_to_labels(u_inner_unnor_1)
#u_inner_unnor_label_dict_1 = label_to_dict(u_inner_unnor_label_1)

#print("MMBO1 with inner step & unnormalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_inner_unnor_1))

#modularity_1_inner_unnor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_unnor_label_1,resolution=1)
#ARI_mbo_1_inner_unnor_1 = adjusted_rand_score(u_inner_unnor_label_1, gt_membership)
#purify_mbo_1_inner_unnor_1 = purity_score(gt_membership, u_inner_unnor_label_1)
#inverse_purify_mbo_1_inner_unnor_1 = inverse_purity_score(gt_membership, u_inner_unnor_label_1)
#NMI_mbo_1_inner_unnor_1 = normalized_mutual_info_score(gt_membership, u_inner_unnor_label_1)
#AMI_mbo_1_inner_unnor_1 = adjusted_mutual_info_score(gt_membership, u_inner_unnor_label_1)


#print(' modularity_1 inner step unnormalized score: ', modularity_1_inner_unnor_1)
#print(' ARI_1 inner step unnormalized score: ', ARI_mbo_1_inner_unnor_1)
#print(' purify for MMBO1 inner step with unnormalized \eta =1 : ', purify_mbo_1_inner_unnor_1)
#print(' inverse purify for MMBO1 inner step with unnormalized \eta =1 : ', inverse_purify_mbo_1_inner_unnor_1)
#print(' NMI for MMBO1 inner step with unnormalized \eta =1 : ', NMI_mbo_1_inner_unnor_1)
#print(' AMI for MMBO1 inner step with unnormalized \eta =1 : ', AMI_mbo_1_inner_unnor_1)



start_time_1_inner_nor_1 = time.time()

# MMBO1 with inner step & sym normalized L_F and gamma=1

#u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, nor_graph_laplacian_1, nor_signless_laplacian_1,dt_inner, tol,target_size_1, inner_step_count)
#u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
#u_inner_nor_label_dict_1 = label_to_dict(u_inner_nor_label_1)

#print("MMBO1 with inner step & normalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_inner_nor_1))

#modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_1,resolution=1)
#ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_membership)
#purify_mbo_1_inner_nor_1 = purity_score(gt_membership, u_inner_nor_label_1)
#inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_membership, u_inner_nor_label_1)
#NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_membership, u_inner_nor_label_1)
#AMI_mbo_1_inner_nor_1 = adjusted_mutual_info_score(gt_membership, u_inner_nor_label_1)


#print(' modularity_1 inner step sym normalized score: ', modularity_1_inner_nor_1)
#print(' ARI_1 inner step sym normalized score: ', ARI_mbo_1_inner_nor_1)
#print(' purify for MMBO1 inner step with sym normalized \eta =1 : ', purify_mbo_1_inner_nor_1)
#print(' inverse purify for MMBO1 inner step with sym normalized \eta =1 : ', inverse_purify_mbo_1_inner_nor_1)
#print(' NMI for MMBO1 inner step with sym normalized \eta =1 : ', NMI_mbo_1_inner_nor_1)
#print(' AMI for MMBO1 inner step with sym normalized \eta =1 : ', AMI_mbo_1_inner_nor_1)


start_time_1_inner_rw_1 = time.time()

# MMBO1 with inner step & random walk L_F & Q_H and gamma=1

#u_inner_rw_1,num_repeat_inner_rw = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, random_walk_nor_lap_1, rw_signless_laplacian_1,dt_inner, tol,target_size_1, inner_step_count)
#u_inner_rw_label_1 = vector_to_labels(u_inner_rw_1)
#u_inner_rw_label_dict_1 = label_to_dict(u_inner_rw_label_1)

#print("MMBO1 with inner step & rw L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_inner_rw_1))

#modularity_1_inner_rw_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_rw_label_1,resolution=1)
#ARI_mbo_1_inner_rw_1 = adjusted_rand_score(u_inner_rw_label_1, gt_membership)
#purify_mbo_1_inner_rw_1 = purity_score(gt_membership, u_inner_rw_label_1)
#inverse_purify_mbo_1_inner_rw_1 = inverse_purity_score(gt_membership, u_inner_rw_label_1)
#NMI_mbo_1_inner_rw_1 = normalized_mutual_info_score(gt_membership, u_inner_rw_label_1)
#AMI_mbo_1_inner_rw_1 = adjusted_mutual_info_score(gt_membership, u_inner_rw_label_1)

#print(' modularity_1 inner step random walk score: ', modularity_1_inner_rw_1)
#print(' ARI_1 inner step random walk score: ', ARI_mbo_1_inner_rw_1)
#print(' purify for MMBO1 inner step with random walk  \eta =1 : ', purify_mbo_1_inner_rw_1)
#print(' inverse purify for MMBO1 inner step with random walk  \eta =1 : ', inverse_purify_mbo_1_inner_rw_1)
#print(' NMI for MMBO1 inner step with random walk  \eta =1 : ', NMI_mbo_1_inner_rw_1)
#print(' AMI for MMBO1 inner step with random walk \eta =1 : ', AMI_mbo_1_inner_rw_1)



start_time_2_inner_B_unnor_1 = time.time()
# MMBO1 with inner step & unnormalized B^+ & B^- and gamma=1

#u_inner_B_unnor_1,num_repeat_inner_B_unnor = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, graph_laplacian_positive_B_1, signless_lap_neg_B_1,dt_inner, tol,target_size_1, inner_step_count)
#u_inner_B_unnor_label_1 = vector_to_labels(u_inner_B_unnor_1)
#u_inner_B_unnor_label_dict_1 = label_to_dict(u_inner_B_unnor_label_1)

#print("MMBO1 with inner step & unnormalized B^+ & B^- and gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_unnor_1))

#modularity_1_inner_B_unnor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_B_unnor_label_1,resolution=1)
#ARI_mbo_1_inner_B_unnor_1 = adjusted_rand_score(u_inner_B_unnor_label_1, gt_membership)
#purify_mbo_1_inner_B_unnor_1 = purity_score(gt_membership, u_inner_B_unnor_label_1)
#inverse_purify_mbo_1_inner_B_unnor_1 = inverse_purity_score(gt_membership, u_inner_B_unnor_label_1)
#NMI_mbo_1_inner_B_unnor_1 = normalized_mutual_info_score(gt_membership, u_inner_B_unnor_label_1)
#AMI_mbo_1_inner_B_unnor_1 = adjusted_mutual_info_score(gt_membership, u_inner_B_unnor_label_1)

#print(' modularity_1 inner step unnormalized B^+ & B^-score: ', modularity_1_inner_B_unnor_1)
#print(' ARI_1 inner step unnormalized B^+ & B^-score: ', ARI_mbo_1_inner_B_unnor_1)
#print(' purify for MMBO1 inner step withunnormalized B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_unnor_1)
#print(' inverse purify for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_unnor_1)
#print(' NMI for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_unnor_1)
#print(' AMI for MMBO1 inner step with unnormalized B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_unnor_1)


start_time_2_inner_B_nor_1 = time.time()
# MMBO1 with inner step & sym normalized B^+ & B^- and gamma=1

#u_inner_B_nor_1,num_repeat_inner_B_nor = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, sym_lap_positive_B_1, sym_signless_lap_negative_B_1,dt_inner, tol,target_size_1, inner_step_count)
#u_inner_B_nor_label_1 = vector_to_labels(u_inner_B_nor_1)
#u_inner_B_nor_label_dict_1 = label_to_dict(u_inner_B_nor_label_1)

#print("MMBO1 with inner step & sym normalized B^+ & B^- gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_nor_1))

#modularity_1_inner_B_nor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_B_nor_label_1,resolution=1)
#ARI_mbo_1_inner_B_nor_1 = adjusted_rand_score(u_inner_B_nor_label_1, gt_membership)
#purify_mbo_1_inner_B_nor_1 = purity_score(gt_membership, u_inner_B_nor_label_1)
#inverse_purify_mbo_1_inner_B_nor_1 = inverse_purity_score(gt_membership, u_inner_B_nor_label_1)
#NMI_mbo_1_inner_B_nor_1 = normalized_mutual_info_score(gt_membership, u_inner_B_nor_label_1)
#AMI_mbo_1_inner_B_nor_1 = adjusted_mutual_info_score(gt_membership, u_inner_B_nor_label_1)

#print(' modularity_1 inner step sym normalized B^+ & B^- score: ', modularity_1_inner_B_nor_1)
#print(' ARI_1 inner step sym normalized B^+ & B^- score: ', ARI_mbo_1_inner_B_nor_1)
#print(' purify for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_nor_1)
#print(' inverse purify for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_nor_1)
#print(' NMI for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_nor_1)
#print(' AMI for MMBO1 inner step with sym normalized B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_nor_1)


start_time_2_inner_B_rw_1 = time.time()
# MMBO1 with inner step & random ealk B^+ & B^- and gamma=1

#u_inner_B_rw_1,num_repeat_inner_B_rw = mbo_modularity_inner_step(num_nodes_1, num_communities, m_1, rw_nor_lap_positive_B_1, rw_signless_lap_negative_B_1,dt_inner, tol,target_size_1, inner_step_count)
#u_inner_B_rw_label_1 = vector_to_labels(u_inner_B_rw_1)
#u_inner_B_rw_label_dict_1 = label_to_dict(u_inner_B_rw_label_1)

#print("MMBO1 with inner step & random ealk B^+ & B^- gamma=1:-- %.3f seconds --" % (time.time() - start_time_2_inner_B_rw_1))

#modularity_1_inner_B_rw_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_B_rw_label_1,resolution=1)
#ARI_mbo_1_inner_B_rw_1 = adjusted_rand_score(u_inner_B_rw_label_1, gt_membership)
#purify_mbo_1_inner_B_rw_1 = purity_score(gt_membership, u_inner_B_rw_label_1)
#inverse_purify_mbo_1_inner_B_rw_1 = inverse_purity_score(gt_membership, u_inner_B_rw_label_1)
#NMI_mbo_1_inner_B_rw_1 = normalized_mutual_info_score(gt_membership, u_inner_B_rw_label_1)
#AMI_mbo_1_inner_B_rw_1 = adjusted_mutual_info_score(gt_membership, u_inner_B_rw_label_1)

#print(' modularity_1 inner step random ealk B^+ & B^- score: ', modularity_1_inner_B_rw_1)
#print(' ARI_1 inner step random ealk B^+ & B^- score: ', ARI_mbo_1_inner_B_rw_1)
#print(' purify for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', purify_mbo_1_inner_B_rw_1)
#print(' inverse purify for MMBO1 inner step with srandom ealk B^+ & B^- \eta =1 : ', inverse_purify_mbo_1_inner_B_rw_1)
#print(' NMI for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', NMI_mbo_1_inner_B_rw_1)
#print(' AMI for MMBO1 inner step with random ealk B^+ & B^- \eta =1 : ', AMI_mbo_1_inner_B_rw_1)



start_time_hu_original = time.time()

# test HU original MBO

#u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes_1, num_communities, m_100,degree_1, dt_inner, nor_graph_laplacian_1, tol,target_size_1 ,inner_step_count) 
#u_hu_label_1 = vector_to_labels(u_hu_vector)
#u_hu_dict_1 = label_to_dict(u_hu_label_1)

#print("HU original MBO:-- %.3f seconds --" % (time.time() - start_time_hu_original))

#modu_hu_original_1 = skn.clustering.modularity(adj_mat_nparray,u_hu_label_1,resolution=1)
#ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_membership)
#purify_hu_original_1 = purity_score(gt_membership, u_hu_label_1)
#inverse_purify_hu_original_1 = inverse_purity_score(gt_membership, u_hu_label_1)
#NMI_hu_original_1 = normalized_mutual_info_score(gt_membership, u_hu_label_1)
#AMI_hu_original_1 = adjusted_mutual_info_score(gt_membership, u_hu_label_1)

#print(' modularity score for HU original MBO: ', modu_hu_original_1)
#print(' ARI for HU original MBO: ', ARI_hu_original_1)
#print(' purify for HU original MBO : ', purify_hu_original_1)
#print(' inverse purify for HU original MBO : ', inverse_purify_hu_original_1)
#print(' NMI for HU original MBO : ', NMI_hu_original_1)
#print(' AMI for HU original MBO : ', AMI_hu_original_1)


start_time_louvain = time.time()

# Louvain algorithm (can setting resolution gamma)
partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)
#print('Louvain:', type(partition_Louvain))
#print('louvain: ',louvain_list)

print("Louvain algorithm:-- %.3f seconds --" % (time.time() - start_time_louvain))

modularity_louvain = skn.clustering.modularity(adj_mat_nparray,louvain_array,resolution=0.5)
ARI_louvain = adjusted_rand_score(louvain_array, gt_membership)
purify_louvain = purity_score(gt_membership, louvain_array)
inverse_purify_louvain = inverse_purity_score(gt_membership, louvain_array)
NMI_louvain = normalized_mutual_info_score(gt_membership, louvain_array)

print('average modularity Louvain score: ', modularity_louvain)
print('average ARI Louvain  score: ', ARI_louvain)
print('average purify for Louvain : ', purify_louvain)
print('average inverse purify for Louvain : ', inverse_purify_louvain)
print('average NMI for Louvain with \eta =1 : ', NMI_louvain)



start_time_CNM = time.time()

# CNM algorithm (can setting resolution gamma)
#partition_CNM = nx_comm.greedy_modularity_communities(G)

#partition_CNM_list = [list(x) for x in partition_CNM]
#print(type(partition_CNM_list))

#partition_CNM_expand = sum(partition_CNM_list, [])

#num_cluster_CNM = []
#for cluster in range(len(partition_CNM_list)):
#    for number_CNM in range(len(partition_CNM_list[cluster])):
#        num_cluster_CNM.append(cluster)

#print(partition_CNM_list)
#CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
#print('CNM: ',CNM_dict)

#CNM_list = list(dict.values(CNM_dict))    #convert a dict to list

#print("CNM algorithm:-- %.3f seconds --" % (time.time() - start_time_CNM))


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

start_time_spectral_clustering = time.time()

# Spectral clustering with k-means
#sc = SpectralClustering(n_clusters=15, affinity='precomputed')
#assignment = sc.fit_predict(adj_mat)

#ass_vec = labels_to_vector(assignment)
#ass_dict = label_to_dict (assignment)

#print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))



## Compute modularity scores

modu_gt = co.modularity(gt_label_dict,G)

#modu_louvain = co.modularity(partition_Louvain,G)
#modu_CNM = co.modularity(CNM_dict,G)
#modu_GN = co.modularity(GN_dict,G)
#modu_sc = co.modularity(ass_dict,G)
#modularity_GN_1 = get_modularity(G,GN_dict)
#modularity_CNM_2 = nx_comm.modularity(G,partition_CNM_list)
#modu_louvain = nx_comm.modularity(G, louvain_list)

print('modularity_gt score:',modu_gt)

#print('modularity_Louvain score:',modu_louvain)
#print('modularity_CNM score:',modu_CNM)
#print('modularity_GN score:',modu_GN)
#print('modularity_GN_1 score:',modularity_GN_1)
#print('modularity_CNM_2 score:',modularity_CNM_2)
#print('modularity_spectral clustering score:',modu_sc)




## Compare ARI 
#ARI_spectral_clustering = adjusted_rand_score(assignment, gt_membership)
#ARI_gn = adjusted_rand_score(partition_GN, gt_membership)
#ARI_louvain = adjusted_rand_score(louvain_list, gt_membership)
#ARI_CNM = adjusted_rand_score(CNM_list, gt_membership)

#print('ARI for spectral clustering: ', ARI_spectral_clustering)
#print('ARI for GN: ', ARI_gn)
#print('ARI for Louvain: ', ARI_louvain)
#print('ARI for CNM: ', ARI_CNM)



# compute purity
#)
#pirify_gn = purity_score(gt_membership, partition_GN)
#purify_louvain = purity_score(gt_membership, louvain_list)
#purify_CNM = purity_score(gt_membership, CNM_list)

#print('purify for spectral clustering: ', purify_spectral_clustering)
#print('purify for GN: ', purify_gn)
#print('purify for Louvain: ', purify_louvain)
#print('purify for CNM: ', purify_CNM)



# compute Inverse Purity
#inverse_purify_spectral_clustering = inverse_purity_score(gt_membership, assignment)
#inverse_purify_gn = inverse_purity_score(gt_membership, partition_GN)
#inverse_purify_louvain = inverse_purity_score(gt_membership, louvain_list)
#inverse_purify_CNM = inverse_purity_score(gt_membership, CNM_list)

#print('inverse purify for spectral clustering: ', inverse_purify_spectral_clustering)
#print('inverse purify for GN: ', inverse_purify_gn)
#print('inverse purify for Louvain: ', inverse_purify_louvain)
#print('inverse purify for CNM: ', inverse_purify_CNM)



# compute Normalized Mutual Information (NMI)
#NMI_spectral_clustering = normalized_mutual_info_score(gt_membership, assignment)
#NMI_gn = normalized_mutual_info_score(gt_membership, partition_GN)
#NMI_louvain = normalized_mutual_info_score(gt_membership, louvain_list)
#NMI_CNM = normalized_mutual_info_score(gt_membership, CNM_list)

#print('AMI for spectral clustering: ', NMI_spectral_clustering)
#print('AMI for GN: ', NMI_gn)
#print('AMI for Louvain: ', NMI_louvain)
#print('AMI for CNM: ', NMI_CNM)



# compute Adjusted Mutual Information (AMI)
#AMI_spectral_clustering = adjusted_mutual_info_score(gt_membership, assignment)
#AMI_gn = adjusted_mutual_info_score(gt_membership, partition_GN)
#AMI_louvain = adjusted_mutual_info_score(gt_membership, louvain_list)
#AMI_CNM = adjusted_mutual_info_score(gt_membership, CNM_list)

#print('AMI for spectral clustering: ', AMI_spectral_clustering)
#print('AMI for GN: ', AMI_gn)
#print('AMI for Louvain: ', AMI_louvain)
#print('AMI for CNM: ', AMI_CNM)


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
#               ARI_louvain, ARI_CNM, ARI_spectral_clustering]


with open('SBM_test.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray)
#    wr.writerow(resultarray)