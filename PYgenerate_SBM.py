from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from scipy.linalg import eigh
from MBO_Network import adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step
from MBO_Network import mbo_modularity_hu_original,MMBO2_preliminary, mbo_modularity_1
from graph_mbo.utils import vector_to_labels, get_initial_state_1,labels_to_vector,to_standard_labels,purity_score,inverse_purity_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import scipy as sp
from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh,eigs
import time
import csv
import sknetwork as skn
from community import community_louvain
from graph_cut_util import generate_initial_value_multiclass



# parameter setting
dt_inner = 1
num_communities = 5
#alg_K = 9
m = 1 * num_communities
tol = 1e-4
inner_step_count = 3
N = 3000
num_nodes_each_cluster = int(N/num_communities)
#print('num_nodes_each_cluster type: ', type(num_nodes_each_cluster))

print('num_communities = 5')

sizes = []

for i in range(num_communities):
    sizes.append(num_nodes_each_cluster)

#print(len(sizes))

all_one_matrix = np.ones((len(sizes),len(sizes)))
diag_matrix = np.diag(np.full(len(sizes),1))
#probs = 0.01 * all_one_matrix + 0.94 * diag_matrix
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
#adj_mat = nx.to_numpy_matrix(G)
adj_mat_nparray = nx.convert_matrix.to_numpy_array(G)
#print('adj type: ', adj_mat_nparray.shape)

#u_init = generate_initial_value_multiclass('rd_equal', n_samples=N, n_class=num_communities)

# Louvain
#start_time_louvain = time.time()
#G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#G = nx.convert_matrix.from_numpy_array(adj_mat_nparray)
#partition_Louvain = community_louvain.best_partition(G, resolution=1)    # returns a dict
#print('partition_Louvain: ', type(partition_Louvain))
#louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#louvain_array = np.asarray(louvain_list)
#print('louvain_array: ', louvain_array.shape)
#louvain_array = to_standard_labels(louvain_array)
#louvain_vec = labels_to_vector(louvain_array)
#print('louvain_vec: ', louvain_vec)
#print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))
#louvain_cluster_list = np.unique(louvain_array)
#print('louvain_cluster_list: ', louvain_cluster_list)
#louvain_cluster = len(louvain_cluster_list)
#print('number of clusters Louvain found: ', louvain_cluster)

#modularity_louvain = skn.clustering.modularity(adj_mat_nparray,louvain_array,resolution=1)

#print('modularity Louvain score: ', modularity_louvain)

#num_communities = louvain_cluster 
#m = 1 * num_communities

num_nodes, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat_nparray, num_communities)
l_mix_sym = sym_graph_lap + sym_signless_lap
l_mix_rw = rw_graph_lap + rw_signless_lap

num_nodes, degree_B, target_size, graph_laplacian_positive_B, sym_lap_positive_B, rw_nor_lap_positive_B, signless_lap_neg_B, sym_signless_lap_negative_B, rw_signless_lap_negative_B = MMBO2_preliminary(adj_mat_nparray, num_communities)
l_mix_rw_B = rw_nor_lap_positive_B + rw_signless_lap_negative_B
l_mix_sym_B = sym_lap_positive_B + sym_signless_lap_negative_B
#print('num_nodes: ', num_nodes)
#print('degree: ', degree.shape)

# Initialize u
#start_time_initialize = time.time()
#u_init = get_initial_state_1(num_nodes, num_communities)
#u_init = generate_initial_value_multiclass('rd_equal', n_samples=N, n_class=num_communities)
#print('u_init: ', u_init)
#time_initialize_u = time.time() - start_time_initialize
#print("compute initialize u:-- %.3f seconds --" % (time_initialize_u))


start_time_l_sym = time.time()
D_sym, V_sym = eigsh(sym_graph_lap, k=m, which='SM')
time_eig_l_sym = time.time() - start_time_l_sym
#print("eigendecomposition L_{sym}:-- %.3f seconds --" % (time_eig_l_sym))
#D_sym = D_sym[1:m+1]
#D_sym = np.where(D_sym > 0, D_sym, 0)
#D_sym = np.insert(D_sym,0,0)
#print('D_sym: ', D_sym)
#print('V_sym: ', V_sym)


D_rw, V_rw = eigsh(rw_graph_lap, k=m, which='SM')
#time_eig_l_sym = time.time() - start_time_l_sym
#print("eigendecomposition L_{sym}:-- %.3f seconds --" % (time_eig_l_sym))
#D_rw = D_rw[1:m+1]
#D_rw = np.where(D_rw > 0, D_rw, 0)
#D_rw = np.insert(D_rw,0,0)
#print('D_rw: ', D_rw)

#D_sym_signless, V_sym_signless = eigh(sym_signless_lap)
#D_sym_signless = D_sym_signless[:m]
#V_sym_signless = V_sym_signless[:,:m]
#V_sym_signless = np.where(V_sym_signless > 1e-2, V_sym_signless, 0)
#D_sym_signless, V_sym_signless = eigsh(sym_signless_lap, k=m, which='SM')

#num_elements = np.unique(V_sym_signless)
#len_num_elements = len(num_elements)
#unique, counts = np.unique(V_sym_signless, return_counts=True)
#num_each_element = dict(zip(unique, counts))
#print('num_elements: ', num_elements)
#print('len_num_elements: ', len_num_elements)
#print('num_each_element: ', num_each_element)
#print('D_sym_signless: ', D_sym_signless)
#print('V_sym_signless: ', V_sym_signless)
#print('type V_sym_signless: ', V_sym_signless.shape)

#print('norm_V_sym_signless: ', norm_V_sym_signless)

#D_sym_signless_B_pos, V_sym_signless_B_pos = eigsh(sym_lap_positive_B, k=m, which='SM')
#D_sym_signless_B_neg, V_sym_signless_B_neg = eigsh(sym_signless_lap_negative_B, k=m, which='SM')

#print('Q_{P_{sym}}: ', D_sym_signless)
#print('Q_{B_{sym}^-}: ', D_sym_signless_B_neg)

start_time_l_mix = time.time()
D_mmbo_sym, V_mmbo_sym = eigsh(l_mix_sym, k=m, which='SM')
time_eig_l_mix = time.time() - start_time_l_mix
#print("eigendecomposition L_{mix}:-- %.3f seconds --" % (time_eig_l_mix))
#D_mmbo_sym = np.insert(D_mmbo_sym,0,0)
#print('D_mmbo_sym: ', D_mmbo_sym)
#print('V_mmbo_sym: ', V_mmbo_sym)


D_mmbo_rw, V_mmbo_rw = eigsh(l_mix_rw, k=m, which='SM')
#time_eig_l_mix = time.time() - start_time_l_mix
#print("eigendecomposition L_{mix}:-- %.3f seconds --" % (time_eig_l_mix))
#D_mmbo_rw = np.insert(D_mmbo_rw,0,0)
#print('D_mmbo_rw: ', D_mmbo_rw)

start_time_l_mix_B = time.time()
D_mmbo_sym_B, V_mmbo_sym_B = eigsh(l_mix_sym_B, k=m, which='SM')
time_eig_l_mix_B = time.time() - start_time_l_mix_B
#print("eigendecomposition L_{mix}:-- %.3f seconds --" % (time_eig_l_mix))
#D_mmbo_sym_B = np.insert(D_mmbo_sym_B,0,0)
#print('D_mmbo_sym_B: ', D_mmbo_sym_B)


D_mmbo_rw_B, V_mmbo_rw_B = eigsh(l_mix_rw_B, k=m, which='SM')
#D_mmbo_rw_B = np.insert(D_mmbo_rw_B,0,0)
#print('D_mmbo_rw_B: ', D_mmbo_rw_B)

#D, V = eigh(graph_laplacian)
#D = D[:30]

#D_sym, V_sym = eigh(sym_graph_lap)
#D_sym = D_sym[:30]
#D_sym = np.where(D_sym > 0, D_sym, 0)
#print('D_sym: ', D_sym)

#D_rw, V_rw = eigh(rw_graph_lap)
#D_rw = D_rw[:30]

#D_mmbo_sym, V_mmbo_sym = eigh(l_mix_sym)
#D_mmbo_sym = D_mmbo_sym[:30]
#print('D_mmbo_sym: ', D_mmbo_sym)

#D_mmbo_sym_B, V_mmbo_sym_B = eigh(l_mix_sym_B)

#D_mmbo_rw, V_mmbo_rw = eigh(l_mix_rw)
#D_mmbo_rw = D_mmbo_rw[:30]
#print('D_mmbo: ', D_mmbo.shape)
#print('V_mmbo: ', V_mmbo.shape)



#D_sym = np.squeeze(D_sym[:m])
#V_sym = V_sym[:,:m]

# Test HU original MBO with symmetric normalized L_F

sum_modularity_hu_sym =0
sum_ARI_hu_original_sym = 0
sum_purify_hu_original_sym =0
sum_inverse_purify_hu_original_sym =0
sum_NMI_hu_original_sym =0
sum_num_iter_HU_sym = 0
sum_time_hu_mbo =0
sum_time_eig_l_sym =0

sum_num_iter_HU_rw =0
sum_modularity_hu_rw =0
sum_ARI_hu_original_rw =0
sum_purify_hu_original_rw =0
sum_inverse_purify_hu_original_rw =0
sum_NMI_hu_original_rw =0

sum_time_MMBO_projection_sym =0
sum_num_repeat_1_nor_Lf_Qh_1 =0
sum_modularity_1_nor_lf_qh =0
sum_ARI_mbo_1_nor_Lf_Qh_1 =0
sum_purify_mbo_1_nor_Lf_Qh_1 =0
sum_inverse_purify_mbo_1_nor_Lf_Qh_1 =0
sum_NMI_mbo_1_nor_Lf_Qh_1 =0
sum_time_eig_l_mix =0

sum_num_repeat_1_nor_Lf_Qh_rw =0
sum_modularity_mmbo_proj_lwq_rw =0
sum_ARI_mmbo_proj_lwq_rw =0
sum_purify_mmbo_proj_lwq_rw =0
sum_inverse_purify_mmbo_proj_lwq_rw =0
sum_NMI_mmbo_proj_lwq_rw =0

sum_num_repeat_mmbo_proj_B_sym =0
sum_modularity_mmbo_proj_B_sym =0
sum_ARI_mmbo_proj_B_sym =0
sum_purify_mmbo_proj_B_sym =0
sum_inverse_purify_mmbo_proj_B_sym =0
sum_NMI_mmbo_proj_B_sym =0

sum_num_repeat_mmbo_proj_B_rw =0
sum_modularity_mmbo_proj_B_rw =0
sum_ARI_mmbo_proj_B_rw =0
sum_purify_mmbo_proj_B_rw =0
sum_inverse_purify_mmbo_proj_B_rw =0
sum_NMI_mmbo_proj_B_rw =0

sum_time_MMBO_inner_step_rw =0
sum_num_repeat_inner_nor =0
sum_modularity_1_inner_nor_1=0
sum_ARI_mbo_1_inner_nor_1 =0
sum_purify_mbo_1_inner_nor_1 =0
sum_inverse_purify_mbo_1_inner_nor_1 =0
sum_NMI_mbo_1_inner_nor_1 =0

sum_num_repeat_inner_rw =0
sum_modularity_1_inner_rw =0
sum_ARI_mbo_1_inner_rw =0
sum_purify_mbo_1_inner_rw =0
sum_inverse_purify_mbo_1_inner_rw =0
sum_NMI_mbo_1_inner_rw =0

sum_num_repeat_inner_nor_B_sym =0
sum_modularity_mmbo_inner_B_sym =0
sum_ARI_mmbo_inner_B_sym =0
sum_purify_mmbo_inner_B_sym =0
sum_inverse_purify_mmbo_inner_B_sym =0
sum_NMI_mmbo_inner_B_sym =0

sum_num_repeat_inner_nor_B_rw =0
sum_modularity_mmbo_inner_B_rw =0
sum_ARI_mmbo_inner_B_rw =0
sum_purify_mmbo_inner_B_rw =0
sum_inverse_purify_mmbo_inner_B_rw =0
sum_NMI_mmbo_inner_B_rw =0

for _ in range(10):
    start_time_initialize = time.time()
    u_init = generate_initial_value_multiclass('rd_equal', n_samples=N, n_class=num_communities)
    time_initialize_u = time.time() - start_time_initialize

    #start_time_hu_original = time.time()
    u_hu_vector, num_iter_HU, HU_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree, dt_inner, u_init,
                                 D_sym, V_sym, tol,inner_step_count, adj_mat_nparray)
    #time_hu_mbo = time.time() - start_time_hu_original
    #time_hu_mbo = time_eig_l_sym + time_initialize_u + time_hu_mbo
    #print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
    #print('the num_iteration of HU method: ', num_iter_HU)

    u_hu_label_1 = vector_to_labels(u_hu_vector)

    #HU_cluster = len(np.unique(u_hu_label_1))
    #print('the cluster Hu method found: ', HU_cluster)

    modu_hu_original_1 = skn.clustering.modularity(adj_mat_nparray,u_hu_label_1,resolution=1)
    ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, gt_membership)
    purify_hu_original_1 = purity_score(gt_membership, u_hu_label_1)
    inverse_purify_hu_original_1 = inverse_purity_score(gt_membership, u_hu_label_1)
    NMI_hu_original_1 = normalized_mutual_info_score(gt_membership, u_hu_label_1)

    #print('modularity score for HU method: ', modu_hu_original_1)
    #print('ARI for HU method: ', ARI_hu_original_1)
    #print('purify for HU method: ', purify_hu_original_1)
    #print('inverse purify for HU method: ', inverse_purify_hu_original_1)
    #print('NMI for HU method: ', NMI_hu_original_1)
    
    #sum_time_eig_l_sym += time_eig_l_sym
    #sum_time_hu_mbo += time_hu_mbo
    sum_num_iter_HU_sym += num_iter_HU 
    sum_modularity_hu_sym += modu_hu_original_1
    sum_ARI_hu_original_sym += ARI_hu_original_1
    sum_purify_hu_original_sym += purify_hu_original_1
    sum_inverse_purify_hu_original_sym += inverse_purify_hu_original_1
    sum_NMI_hu_original_sym += NMI_hu_original_1

    # HU's method --rw
    u_hu_vector_rw, num_iter_HU_rw, HU_modularity_list_rw = mbo_modularity_hu_original(num_nodes, num_communities, m, degree, dt_inner, u_init,
                                 D_rw, V_rw, tol,inner_step_count, adj_mat_nparray)
    #time_hu_mbo = time.time() - start_time_hu_original
    #time_hu_mbo = time_eig_l_sym + time_initialize_u + time_hu_mbo
    #print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
    #print('the num_iteration of HU method: ', num_iter_HU)

    u_hu_label_rw = vector_to_labels(u_hu_vector_rw)

    #HU_cluster = len(np.unique(u_hu_label_1))
    #print('the cluster Hu method found: ', HU_cluster)

    modu_hu_original_rw = skn.clustering.modularity(adj_mat_nparray,u_hu_label_rw,resolution=1)
    ARI_hu_original_rw = adjusted_rand_score(u_hu_label_rw, gt_membership)
    purify_hu_original_rw = purity_score(gt_membership, u_hu_label_rw)
    inverse_purify_hu_original_rw = inverse_purity_score(gt_membership, u_hu_label_rw)
    NMI_hu_original_rw = normalized_mutual_info_score(gt_membership, u_hu_label_rw)

    #print('HU method --random walk')
    #print('modularity score for HU method: ', modu_hu_original_rw)
    #print('ARI for HU method: ', ARI_hu_original_rw)
    #print('purify for HU method: ', purify_hu_original_rw)
    #print('inverse purify for HU method: ', inverse_purify_hu_original_rw)
    #print('NMI for HU method: ', NMI_hu_original_rw)
    
    #sum_time_eig_l_sym += time_eig_l_sym
    #sum_time_hu_mbo += time_hu_mbo
    sum_num_iter_HU_rw += num_iter_HU_rw 
    sum_modularity_hu_rw += modu_hu_original_rw
    sum_ARI_hu_original_rw += ARI_hu_original_rw
    sum_purify_hu_original_rw += purify_hu_original_rw
    sum_inverse_purify_hu_original_rw += inverse_purify_hu_original_rw
    sum_NMI_hu_original_rw += NMI_hu_original_rw

    
    # MMBO projection l_sym
    #start_time_1_nor_Lf_Qh_1 = time.time()
    u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1, MMBO_projection_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree, u_init, 
                                            D_mmbo_sym, V_mmbo_sym, tol, adj_mat_nparray)
    #time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
    #time_MMBO_projection_sym = time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym
    #print("MMBO using projection with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
    #print('the number of MBO iteration for MMBO using projection with L_W&P: ', num_repeat_1_nor_Lf_Qh_1)

    u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
    modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat_nparray ,u_1_nor_Lf_Qh_individual_label_1,resolution=1)
    ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_membership)
    purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
    inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
    NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)

    #modularity_MMBO_projection_sym_list.append(modularity_1_nor_lf_qh)
    #print('modularity for MMBO using projection with L_W&P: ', modularity_1_nor_lf_qh)
    #print('ARI for MMBO using projection with L_W&P: ', ARI_mbo_1_nor_Lf_Qh_1)
    #print('purify for MMBO using projection with L_W&P: ', purify_mbo_1_nor_Lf_Qh_1)
    #print('inverse purify for MMBO using projection with L_W&P: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
    #print('NMI for MMBO using projection with L_W&P: ', NMI_mbo_1_nor_Lf_Qh_1)

#    sum_time_eig_l_mix += time_eig_l_mix
    #sum_time_MMBO_projection_sym += time_MMBO_projection_sym
    sum_num_repeat_1_nor_Lf_Qh_1 += num_repeat_1_nor_Lf_Qh_1 
    sum_modularity_1_nor_lf_qh += modularity_1_nor_lf_qh
    sum_ARI_mbo_1_nor_Lf_Qh_1 += ARI_mbo_1_nor_Lf_Qh_1
    sum_purify_mbo_1_nor_Lf_Qh_1 += purify_mbo_1_nor_Lf_Qh_1
    sum_inverse_purify_mbo_1_nor_Lf_Qh_1 += inverse_purify_mbo_1_nor_Lf_Qh_1
    sum_NMI_mbo_1_nor_Lf_Qh_1 += NMI_mbo_1_nor_Lf_Qh_1

    # MMBO projection l_rw
    u_1_nor_Lf_Qh_individual_rw,num_repeat_1_nor_Lf_Qh_rw, MMBO_projection_rw_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree, u_init, 
                                            D_mmbo_rw, V_mmbo_rw, tol, adj_mat_nparray)
    #time_MMBO_projection_rw = time.time() - start_time_1_nor_Lf_Qh_rw
    #print("MMBO using projection with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_rw))
    #print('the number of MBO iteration for MMBO using projection with L_W&P_rw: ', num_repeat_1_nor_Lf_Qh_rw)

    u_1_nor_Lf_Qh_individual_label_rw = vector_to_labels(u_1_nor_Lf_Qh_individual_rw)
    modularity_mmbo_proj_lwq_rw = skn.clustering.modularity(adj_mat_nparray,u_1_nor_Lf_Qh_individual_label_rw,resolution=1)
    ARI_mmbo_proj_lwq_rw = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_rw, gt_membership)
    purify_mmbo_proj_lwq_rw = purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_rw)
    inverse_purify_mmbo_proj_lwq_rw = inverse_purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_rw)
    NMI_mmbo_proj_lwq_rw = normalized_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_rw)

    sum_num_repeat_1_nor_Lf_Qh_rw += num_repeat_1_nor_Lf_Qh_rw 
    sum_modularity_mmbo_proj_lwq_rw += modularity_mmbo_proj_lwq_rw
    sum_ARI_mmbo_proj_lwq_rw += ARI_mmbo_proj_lwq_rw
    sum_purify_mmbo_proj_lwq_rw += purify_mmbo_proj_lwq_rw
    sum_inverse_purify_mmbo_proj_lwq_rw += inverse_purify_mmbo_proj_lwq_rw
    sum_NMI_mmbo_proj_lwq_rw += NMI_mmbo_proj_lwq_rw

    # MMBO projection B_sym
    u_mmbo_proj_B_sym, num_repeat_mmbo_proj_B_sym, MMBO_projection_modularity_list_B = mbo_modularity_1(num_nodes,num_communities, m, degree, u_init, 
                                            D_mmbo_sym_B, V_mmbo_sym_B, tol, adj_mat_nparray)
    #time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
    #print("MMBO using projection with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
    #print('the number of MBO iteration for MMBO using projection with L_B_sym: ', num_repeat_mmbo_proj_B_sym)

    u_mmbo_proj_B_sym_label = vector_to_labels(u_mmbo_proj_B_sym)
    modularity_mmbo_proj_B_sym = skn.clustering.modularity(adj_mat_nparray,u_mmbo_proj_B_sym_label,resolution=1)
    ARI_mmbo_proj_B_sym = adjusted_rand_score(u_mmbo_proj_B_sym_label, gt_membership)
    purify_mmbo_proj_B_sym = purity_score(gt_membership, u_mmbo_proj_B_sym_label)
    inverse_purify_mmbo_proj_B_sym = inverse_purity_score(gt_membership, u_mmbo_proj_B_sym_label)
    NMI_mmbo_proj_B_sym = normalized_mutual_info_score(gt_membership, u_mmbo_proj_B_sym_label)

    sum_num_repeat_mmbo_proj_B_sym += num_repeat_mmbo_proj_B_sym 
    sum_modularity_mmbo_proj_B_sym += modularity_mmbo_proj_B_sym
    sum_ARI_mmbo_proj_B_sym += ARI_mmbo_proj_B_sym
    sum_purify_mmbo_proj_B_sym += purify_mmbo_proj_B_sym
    sum_inverse_purify_mmbo_proj_B_sym += inverse_purify_mmbo_proj_B_sym
    sum_NMI_mmbo_proj_B_sym += NMI_mmbo_proj_B_sym

    # MMBO projection B_rw
    u_mmbo_proj_B_rw, num_repeat_mmbo_proj_B_rw, MMBO_projection_B_rw_modularity_list = mbo_modularity_1(num_nodes,num_communities, m, degree, u_init, 
                                            D_mmbo_rw_B, V_mmbo_rw_B, tol, adj_mat_nparray)
    #time_MMBO_projection_rw = time.time() - start_time_1_nor_Lf_Qh_rw
    #print("MMBO using projection with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_rw))
    #print('the number of MBO iteration for MMBO using projection with L_B_rw: ', num_repeat_mmbo_proj_B_rw)

    u_mmbo_proj_B_rw_label = vector_to_labels(u_mmbo_proj_B_rw)
    modularity_mmbo_proj_B_rw = skn.clustering.modularity(adj_mat_nparray,u_mmbo_proj_B_rw_label,resolution=1)
    ARI_mmbo_proj_B_rw = adjusted_rand_score(u_mmbo_proj_B_rw_label, gt_membership)
    purify_mmbo_proj_B_rw = purity_score(gt_membership, u_mmbo_proj_B_rw_label)
    inverse_purify_mmbo_proj_B_rw = inverse_purity_score(gt_membership, u_mmbo_proj_B_rw_label)
    NMI_mmbo_proj_B_rw = normalized_mutual_info_score(gt_membership, u_mmbo_proj_B_rw_label)

    sum_num_repeat_mmbo_proj_B_rw += num_repeat_mmbo_proj_B_rw 
    sum_modularity_mmbo_proj_B_rw += modularity_mmbo_proj_B_rw
    sum_ARI_mmbo_proj_B_rw += ARI_mmbo_proj_B_rw
    sum_purify_mmbo_proj_B_rw += purify_mmbo_proj_B_rw
    sum_inverse_purify_mmbo_proj_B_rw += inverse_purify_mmbo_proj_B_rw
    sum_NMI_mmbo_proj_B_rw += NMI_mmbo_proj_B_rw


    # MMBO using finite difference L_sym
    #start_time_1_inner_nor_1 = time.time()
    u_inner_nor_1,num_repeat_inner_nor, MMBO_inner_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree, dt_inner, u_init, 
                                        D_mmbo_sym_B, V_mmbo_sym_B, tol, inner_step_count, adj_mat_nparray)
    #time_MMBO_inner_step_rw = time.time() - start_time_1_inner_nor_1
    #time_MMBO_inner_step_rw = time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
    #print('the number of MBO iteration for MMBO using inner step with L_W&P: ',num_repeat_inner_nor)

    u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
    modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_1,resolution=1)
    ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_membership)
    purify_mbo_1_inner_nor_1 = purity_score(gt_membership, u_inner_nor_label_1)
    inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_membership, u_inner_nor_label_1)
    NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_membership, u_inner_nor_label_1)

    #modularity_MMBO_inner_sym_list.append(modularity_1_inner_nor_1)
    
    #print('modularity for MMBO using inner step with L_W&P: ', modularity_1_inner_nor_1)
    #print('ARI for MMBO using inner step with L_W&P: ', ARI_mbo_1_inner_nor_1)
    #print('purify for MMBO using inner step with L_W&P: ', purify_mbo_1_inner_nor_1)
    #print('inverse purify for MMBO using inner step with L_W&P: ', inverse_purify_mbo_1_inner_nor_1)
    #print('NMI for MMBO using inner step with L_W&P: ', NMI_mbo_1_inner_nor_1)

    #sum_time_eig_l_mix += time_eig_l_mix
    #sum_time_MMBO_inner_step_rw += time_MMBO_inner_step_rw
    sum_num_repeat_inner_nor += num_repeat_inner_nor 
    sum_modularity_1_inner_nor_1 += modularity_1_inner_nor_1
    sum_ARI_mbo_1_inner_nor_1 += ARI_mbo_1_inner_nor_1
    sum_purify_mbo_1_inner_nor_1 += purify_mbo_1_inner_nor_1
    sum_inverse_purify_mbo_1_inner_nor_1 += inverse_purify_mbo_1_inner_nor_1
    sum_NMI_mbo_1_inner_nor_1 += NMI_mbo_1_inner_nor_1

    # MMBO using finite difference L_rw
    u_inner_nor_rw, num_repeat_inner_rw, MMBO_inner_rw_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree, dt_inner, u_init, 
                                        D_mmbo_rw, V_mmbo_rw, tol, inner_step_count, adj_mat_nparray)
    #time_MMBO_inner_step_rw = time.time() - start_time_1_inner_nor_rw
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
    #print('the number of MBO iteration for MMBO using inner step with L_W&P_rw: ',num_repeat_inner_rw)

    u_inner_nor_label_rw = vector_to_labels(u_inner_nor_rw)
    modularity_mmbo_inner_lwq_rw = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_rw,resolution=1)
    ARI_mmbo_inner_lwq_rw = adjusted_rand_score(u_inner_nor_label_rw, gt_membership)
    purify_mmbo_inner_lwq_rw = purity_score(gt_membership, u_inner_nor_label_rw)
    inverse_purify_mmbo_inner_lwq_rw = inverse_purity_score(gt_membership, u_inner_nor_label_rw)
    NMI_mmbo_inner_lwq_rw = normalized_mutual_info_score(gt_membership, u_inner_nor_label_rw)

    #modularity_MMBO_inner_sym_list.append(modularity_1_inner_nor_1)
    #print('modularity for MMBO using inner step with L_W&P_rw: ', modularity_mmbo_inner_lwq_rw)
    #print('ARI for MMBO using inner step with L_W&P_rw: ', ARI_mmbo_inner_lwq_rw)
    #print('purify for MMBO using inner step with L_W&P_rw: ', purify_mmbo_inner_lwq_rw)
    #print('inverse purify for MMBO using inner step with L_W&P_rw: ', inverse_purify_mmbo_inner_lwq_rw)
    #print('NMI for MMBO using inner step with L_W&P_rw: ', NMI_mmbo_inner_lwq_rw)

    sum_num_repeat_inner_rw += num_repeat_inner_rw 
    sum_modularity_1_inner_rw += modularity_mmbo_inner_lwq_rw
    sum_ARI_mbo_1_inner_rw += ARI_mmbo_inner_lwq_rw
    sum_purify_mbo_1_inner_rw += purify_mmbo_inner_lwq_rw
    sum_inverse_purify_mbo_1_inner_rw += inverse_purify_mmbo_inner_lwq_rw
    sum_NMI_mbo_1_inner_rw += NMI_mmbo_inner_lwq_rw

    # MMBO using finite difference B_sym
    u_inner_nor_B_sym, num_repeat_inner_nor_B_sym, MMBO_inner_modularity_list_B = mbo_modularity_inner_step(num_nodes, num_communities, m,degree, dt_inner, u_init, 
                                    D_mmbo_sym_B, V_mmbo_sym_B, tol, inner_step_count, adj_mat_nparray)
    #time_MMBO_inner_step_rw = time.time() - start_time_1_inner_nor_1
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
    #print('the number of MBO iteration for MMBO using inner step with L_B_sym: ',num_repeat_inner_nor_B_sym)

    u_inner_nor_label_B_sym = vector_to_labels(u_inner_nor_B_sym)
    modularity_mmbo_inner_B_sym = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_B_sym,resolution=1)
    ARI_mmbo_inner_B_sym = adjusted_rand_score(u_inner_nor_label_B_sym, gt_membership)
    purify_mmbo_inner_B_sym = purity_score(gt_membership, u_inner_nor_label_B_sym)
    inverse_purify_mmbo_inner_B_sym = inverse_purity_score(gt_membership, u_inner_nor_label_B_sym)
    NMI_mmbo_inner_B_sym = normalized_mutual_info_score(gt_membership, u_inner_nor_label_B_sym)

    sum_num_repeat_inner_nor_B_sym += num_repeat_inner_nor_B_sym 
    sum_modularity_mmbo_inner_B_sym += modularity_mmbo_inner_B_sym
    sum_ARI_mmbo_inner_B_sym += ARI_mmbo_inner_B_sym
    sum_purify_mmbo_inner_B_sym += purify_mmbo_inner_B_sym
    sum_inverse_purify_mmbo_inner_B_sym += inverse_purify_mmbo_inner_B_sym
    sum_NMI_mmbo_inner_B_sym += NMI_mmbo_inner_B_sym

    # MMBO using finite difference B_rw
    u_inner_nor_B_rw, num_repeat_inner_B_rw, MMBO_inner_B_rw_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree, dt_inner, u_init, 
                                    D_mmbo_rw_B, V_mmbo_rw_B, tol, inner_step_count, adj_mat_nparray)
    #time_MMBO_inner_step_rw = time.time() - start_time_1_inner_nor_rw
    #print("MMBO using inner step with L_W&P:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_inner_step_rw))
    #print('the number of MBO iteration for MMBO using inner step with L_B_rw: ',num_repeat_inner_B_rw)

    u_inner_nor_label_B_rw = vector_to_labels(u_inner_nor_B_rw)
    modularity_mmbo_inner_B_rw = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_B_rw,resolution=1)
    ARI_mmbo_inner_B_rw = adjusted_rand_score(u_inner_nor_label_B_rw, gt_membership)
    purify_mmbo_inner_B_rw = purity_score(gt_membership, u_inner_nor_label_B_rw)
    inverse_purify_mmbo_inner_B_rw = inverse_purity_score(gt_membership, u_inner_nor_label_B_rw)
    NMI_mmbo_inner_B_rw = normalized_mutual_info_score(gt_membership, u_inner_nor_label_B_rw)

    sum_num_repeat_inner_nor_B_rw += num_repeat_inner_B_rw 
    sum_modularity_mmbo_inner_B_rw += modularity_mmbo_inner_B_rw
    sum_ARI_mmbo_inner_B_rw += ARI_mmbo_inner_B_rw
    sum_purify_mmbo_inner_B_rw += purify_mmbo_inner_B_rw
    sum_inverse_purify_mmbo_inner_B_rw += inverse_purify_mmbo_inner_B_rw
    sum_NMI_mmbo_inner_B_rw += NMI_mmbo_inner_B_rw

print('MMBO using finite difference L_sym')
#average_sum_time_eig_l_mix = sum_time_eig_l_mix / 10
#average_time_MMBO_inner_step = sum_time_MMBO_inner_step_rw / 10
average_num_iter_MMBO_inner_step = sum_num_repeat_inner_nor / 10
average_modularity_MMBO_inner_step = sum_modularity_1_inner_nor_1 / 10
average_ARI_MMBO_inner_step = sum_ARI_mbo_1_inner_nor_1 / 10
average_purify_MMBO_inner_step = sum_purify_mbo_1_inner_nor_1 / 10
average_inverse_purify_MMBO_inner_step = sum_inverse_purify_mbo_1_inner_nor_1 / 10
average_NMI_MMBO_inner_step = sum_NMI_mbo_1_inner_nor_1 / 10

#print('average_average_sum_time_eig_l_mix: ', average_sum_time_eig_l_mix)
#print('average_time_MMBO_inner_step: ', average_time_MMBO_inner_step)
print('average_num_iter_MMBO_inner_step: ', average_num_iter_MMBO_inner_step)
print('average_modularity_MMBO_inner_step: ', average_modularity_MMBO_inner_step)
print('average_ARI_MMBO_inner_step: ', average_ARI_MMBO_inner_step)
print('average_purify_MMBO_inner_step: ', average_purify_MMBO_inner_step)
print('average_inverse_purify_MMBO_inner_step: ', average_inverse_purify_MMBO_inner_step)
print('average_NMI_MMBO_inner_step: ', average_NMI_MMBO_inner_step)


print('MMBO using finite difference L_rw')
average_num_iter_MMBO_inner_step_rw = sum_num_repeat_inner_rw / 10
average_modularity_MMBO_inner_step_rw = sum_modularity_1_inner_rw / 10
average_ARI_MMBO_inner_step_rw = sum_ARI_mbo_1_inner_rw / 10
average_purify_MMBO_inner_step_rw = sum_purify_mbo_1_inner_rw / 10
average_inverse_purify_MMBO_inner_step_rw = sum_inverse_purify_mbo_1_inner_rw / 10
average_NMI_MMBO_inner_step_rw = sum_NMI_mbo_1_inner_rw / 10

print('average_num_iter_MMBO_inner_step: ', average_num_iter_MMBO_inner_step_rw)
print('average_modularity_MMBO_inner_step: ', average_modularity_MMBO_inner_step_rw)
print('average_ARI_MMBO_inner_step: ', average_ARI_MMBO_inner_step_rw)
print('average_purify_MMBO_inner_step: ', average_purify_MMBO_inner_step_rw)
print('average_inverse_purify_MMBO_inner_step: ', average_inverse_purify_MMBO_inner_step_rw)
print('average_NMI_MMBO_inner_step: ', average_NMI_MMBO_inner_step_rw)


print('MMBO using finite difference B_sym')
average_num_iter_MMBO_inner_step_B_sym = sum_num_repeat_inner_nor_B_sym / 10
average_modularity_MMBO_inner_step_B_sym = sum_modularity_mmbo_inner_B_sym / 10
average_ARI_MMBO_inner_step_B_sym = sum_ARI_mmbo_inner_B_sym / 10
average_purify_MMBO_inner_step_B_sym = sum_purify_mmbo_inner_B_sym / 10
average_inverse_purify_MMBO_inner_step_B_sym = sum_inverse_purify_mmbo_inner_B_sym / 10
average_NMI_MMBO_inner_step_B_sym = sum_NMI_mmbo_inner_B_sym / 10

print('average_num_iter_MMBO_inner_step: ', average_num_iter_MMBO_inner_step_B_sym)
print('average_modularity_MMBO_inner_step: ', average_modularity_MMBO_inner_step_B_sym)
print('average_ARI_MMBO_inner_step: ', average_ARI_MMBO_inner_step_B_sym)
print('average_purify_MMBO_inner_step: ', average_purify_MMBO_inner_step_B_sym)
print('average_inverse_purify_MMBO_inner_step: ', average_inverse_purify_MMBO_inner_step_B_sym)
print('average_NMI_MMBO_inner_step: ', average_NMI_MMBO_inner_step_B_sym)


print('MMBO using finite difference B_rw')
average_num_iter_MMBO_inner_step_B_rw = sum_num_repeat_inner_nor_B_rw / 10
average_modularity_MMBO_inner_step_B_rw = sum_modularity_mmbo_inner_B_rw / 10
average_ARI_MMBO_inner_step_B_rw = sum_ARI_mmbo_inner_B_rw / 10
average_purify_MMBO_inner_step_B_rw = sum_purify_mmbo_inner_B_rw / 10
average_inverse_purify_MMBO_inner_step_B_rw = sum_inverse_purify_mmbo_inner_B_rw / 10
average_NMI_MMBO_inner_step_B_rw = sum_NMI_mmbo_inner_B_rw / 10

print('average_num_iter_MMBO_inner_step: ', average_num_iter_MMBO_inner_step_B_rw)
print('average_modularity_MMBO_inner_step: ', average_modularity_MMBO_inner_step_B_rw)
print('average_ARI_MMBO_inner_step: ', average_ARI_MMBO_inner_step_B_rw)
print('average_purify_MMBO_inner_step: ', average_purify_MMBO_inner_step_B_rw)
print('average_inverse_purify_MMBO_inner_step: ', average_inverse_purify_MMBO_inner_step_B_rw)
print('average_NMI_MMBO_inner_step: ', average_NMI_MMBO_inner_step_B_rw)


print('MMBO using projection L_sym')
#average_sum_time_eig_l_mix = sum_time_eig_l_mix / 10
#average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 10
average_num_iter_MMBO_projection_sym = sum_num_repeat_1_nor_Lf_Qh_1 / 10
average_modularity_MMBO_projection_sym = sum_modularity_1_nor_lf_qh / 10
average_ARI_MMBO_projection_sym = sum_ARI_mbo_1_nor_Lf_Qh_1 / 10
average_purify_MMBO_projection_sym = sum_purify_mbo_1_nor_Lf_Qh_1 / 10
average_inverse_purify_MMBO_projection_sym = sum_inverse_purify_mbo_1_nor_Lf_Qh_1 / 10
average_NMI_MMBO_projection_sym = sum_NMI_mbo_1_nor_Lf_Qh_1 / 10

#print('average_average_sum_time_eig_l_mix: ', average_sum_time_eig_l_mix)
#print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_sym)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_sym)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_sym)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_sym)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_sym)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_sym)


print('MMBO using projection L_rw')
#average_sum_time_eig_l_mix = sum_time_eig_l_mix / 10
#average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 10
average_num_iter_MMBO_projection_rw = sum_num_repeat_1_nor_Lf_Qh_rw / 10
average_modularity_MMBO_projection_rw = sum_modularity_mmbo_proj_lwq_rw / 10
average_ARI_MMBO_projection_rw = sum_ARI_mmbo_proj_lwq_rw / 10
average_purify_MMBO_projection_rw = sum_purify_mmbo_proj_lwq_rw / 10
average_inverse_purify_MMBO_projection_rw = sum_inverse_purify_mmbo_proj_lwq_rw / 10
average_NMI_MMBO_projection_rw = sum_NMI_mmbo_proj_lwq_rw / 10

#print('average_average_sum_time_eig_l_mix: ', average_sum_time_eig_l_mix)
#print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_rw)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_rw)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_rw)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_rw)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_rw)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_rw)


print('MMBO using projection B_sym')
#average_sum_time_eig_l_mix = sum_time_eig_l_mix / 10
#average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 10
average_num_iter_MMBO_projection_B_sym = sum_num_repeat_mmbo_proj_B_sym / 10
average_modularity_MMBO_projection_B_sym = sum_modularity_mmbo_proj_B_sym / 10
average_ARI_MMBO_projection_B_sym = sum_ARI_mmbo_proj_B_sym / 10
average_purify_MMBO_projection_B_sym = sum_purify_mmbo_proj_B_sym / 10
average_inverse_purify_MMBO_projection_B_sym = sum_inverse_purify_mmbo_proj_B_sym / 10
average_NMI_MMBO_projection_B_sym = sum_NMI_mmbo_proj_B_sym / 10

#print('average_average_sum_time_eig_l_mix: ', average_sum_time_eig_l_mix)
#print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_B_sym)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_B_sym)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_B_sym)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_B_sym)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_B_sym)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_B_sym)


print('MMBO using projection B_rw')
#average_sum_time_eig_l_mix = sum_time_eig_l_mix / 10
#average_time_MMBO_projection_sym = sum_time_MMBO_projection_sym / 10
average_num_iter_MMBO_projection_B_rw = sum_num_repeat_mmbo_proj_B_rw / 10
average_modularity_MMBO_projection_B_rw = sum_modularity_mmbo_proj_B_rw / 10
average_ARI_MMBO_projection_B_rw = sum_ARI_mmbo_proj_B_rw / 10
average_purify_MMBO_projection_B_rw = sum_purify_mmbo_proj_B_rw / 10
average_inverse_purify_MMBO_projection_B_rw = sum_inverse_purify_mmbo_proj_B_rw / 10
average_NMI_MMBO_projection_B_rw = sum_NMI_mmbo_proj_B_rw / 10

#print('average_average_sum_time_eig_l_mix: ', average_sum_time_eig_l_mix)
#print('average_time_MMBO_projection_sym: ', average_time_MMBO_projection_sym)
print('average_num_iter_MMBO_projection_sym: ', average_num_iter_MMBO_projection_B_rw)
print('average_modularity_MMBO_projection_sym: ', average_modularity_MMBO_projection_B_rw)
print('average_ARI_MMBO_projection_sym: ', average_ARI_MMBO_projection_B_rw)
print('average_purify_MMBO_projection_sym: ', average_purify_MMBO_projection_B_rw)
print('average_inverse_purify_MMBO_projection_sym: ', average_inverse_purify_MMBO_projection_B_rw)
print('average_NMI_MMBO_projection_sym: ', average_NMI_MMBO_projection_B_rw)


print('HU method L_sym')
#average_time_eig_l_sym = sum_time_eig_l_sym / 10
#average_time_hu_mbo = sum_time_hu_mbo / 10
average_num_iter_HU_sym = sum_num_iter_HU_sym / 10
average_modularity_hu_sym = sum_modularity_hu_sym / 10
average_ARI_hu_original_sym = sum_ARI_hu_original_sym / 10
average_purify_hu_original_sym = sum_purify_hu_original_sym / 10
average_inverse_purify_hu_original_sym = sum_inverse_purify_hu_original_sym / 10
average_NMI_hu_original_sym = sum_NMI_hu_original_sym / 10

#print('average_time_eig_l_sym: ', average_time_eig_l_sym)
#print('average_time_hu_mbo: ', average_time_hu_mbo)
print('average_num_iter_HU_sym: ', average_num_iter_HU_sym)
print('average_modularity_hu_sym: ', average_modularity_hu_sym)
print('average_ARI_hu_original_sym: ', average_ARI_hu_original_sym)
print('average_purify_hu_original_sym: ', average_purify_hu_original_sym)
print('average_inverse_purify_hu_original_sym: ', average_inverse_purify_hu_original_sym)
print('average_NMI_hu_original_sym: ', average_NMI_hu_original_sym)


print('HU method L_rw')
#average_time_eig_l_sym = sum_time_eig_l_sym / 10
#average_time_hu_mbo = sum_time_hu_mbo / 10
average_num_iter_HU_rw = sum_num_iter_HU_rw / 10
average_modularity_hu_rw = sum_modularity_hu_rw / 10
average_ARI_hu_original_rw = sum_ARI_hu_original_rw / 10
average_purify_hu_original_rw = sum_purify_hu_original_rw / 10
average_inverse_purify_hu_original_rw = sum_inverse_purify_hu_original_rw / 10
average_NMI_hu_original_rw = sum_NMI_hu_original_rw / 10

#print('average_time_eig_l_sym: ', average_time_eig_l_sym)
#print('average_time_hu_mbo: ', average_time_hu_mbo)
print('average_num_iter_HU_sym: ', average_num_iter_HU_rw)
print('average_modularity_hu_sym: ', average_modularity_hu_rw)
print('average_ARI_hu_original_sym: ', average_ARI_hu_original_rw)
print('average_purify_hu_original_sym: ', average_purify_hu_original_rw)
print('average_inverse_purify_hu_original_sym: ', average_inverse_purify_hu_original_rw)
print('average_NMI_hu_original_sym: ', average_NMI_hu_original_rw)



#u_hu_vector_rw, num_iter_HU_rw, HU_modularity_list_rw = mbo_modularity_hu_original(num_nodes, alg_K, m, degree, dt_inner, u_init,
#                             D_rw, V_rw, tol,inner_step_count, adj_mat_nparray)

#time_hu_mbo = time.time() - start_time_hu_original
#print("total running time of HU method:-- %.3f seconds --" % (time_eig_l_sym + time_initialize_u + time_hu_mbo))
#print('the num_iteration of HU method: ', num_iter_HU)

#u_hu_label_rw = vector_to_labels(u_hu_vector_rw)
#modu_hu_original_rw = skn.clustering.modularity(adj_mat_nparray,u_hu_label_rw,resolution=1)
#print('modularity score for HU method: ', modu_hu_original_rw)


m_range = list(range(5, 21))
modularity_MMBO_projection_sym_list =[]
modularity_MMBO_inner_sym_list = []
modularity_hu_sym_list=[]
modularity_MMBO_inner_sym_B_list =[]
modularity_MMBO_projection_sym_B_list= []

#u_initial = u_init.copy()

#eigval_hu = D_sym.copy()
#eigvec_hu = V_sym.copy()
#eigval_mmbo_sym = D_mmbo_sym.copy()
#eigvec_mmbo_sym = V_mmbo_sym.copy()
#eigval_mmbo_B_sym = D_mmbo_sym_B.copy()
#eigvec_mmbo_B_sym = V_mmbo_sym_B.copy()

#for i in m_range:
#    print('number of eigenvalues to use: ', i)
#    m = i
#    D_sym, V_sym = eigsh(sym_graph_lap, k=m, which='SM')
        
#    D_mmbo_sym, V_mmbo_sym = eigsh(l_mix_sym, k=m, which='SM')
    #print('D_mmbo_sym: ', D_mmbo_sym.shape)
    #print('V_mmbo_sym: ', V_mmbo_sym.shape)

#    D_mmbo_sym_B, V_mmbo_sym_B = eigsh(l_mix_sym_B, k=m, which='SM')

#D_sym = eigval_hu[:i]
#V_sym = eigvec_hu[:,:i]
#print('D_sym: ', D_sym.shape)
#print('V_sym: ', V_sym.shape)

#D_mmbo_sym = eigval_mmbo_sym[:i]
#V_mmbo_sym = eigvec_mmbo_sym[:,:i]

#D_mmbo_sym_B = eigval_mmbo_B_sym[:i]
#V_mmbo_sym_B = eigvec_mmbo_B_sym[:,:i]

#    u_hu_vector, num_iter_HU, HU_modularity_list = mbo_modularity_hu_original(num_nodes, num_communities, m, degree, dt_inner, u_init,
#                                D_sym, V_sym, tol,inner_step_count, adj_mat_nparray)
#    u_hu_label_1 = vector_to_labels(u_hu_vector)
#    HU_cluster = len(np.unique(u_hu_label_1))
#    print('the cluster Hu method found: ', HU_cluster)
#    modu_hu_original_1 = skn.clustering.modularity(adj_mat_nparray,u_hu_label_1,resolution=1)
#    modularity_hu_sym_list.append(modu_hu_original_1)
#    print('modularity score for HU method: ', modu_hu_original_1)


#D_sym = D_sym[1:m+1]
#V_sym = V_sym[:, 1:m+1]
## Test MMBO using the projection on the eigenvectors with symmetric normalized L_F & Q_H
#start_time_1_nor_Lf_Qh_1 = time.time()
#u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1, MMBO_projection_modularity_list = mbo_modularity_1(num_nodes, num_communities, m, degree, u_init, 
#                                        D_mmbo_sym, V_mmbo_sym, tol, adj_mat_nparray)
#time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1                                                
#print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
#    print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)

#u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

#MMBO_projection_cluster_list = np.unique(u_1_nor_Lf_Qh_individual_label_1)
#print('MMBO_projection_cluster list: ', MMBO_projection_cluster_list)
#    MMBO_projection_cluster = len(MMBO_projection_cluster_list)
#    print('the cluster MMBO using projection found: ', MMBO_projection_cluster)

#modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat_nparray,u_1_nor_Lf_Qh_individual_label_1,resolution=1)
#ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_membership)
#purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
#inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
#NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)

  #  modularity_MMBO_projection_sym_list.append(modularity_1_nor_lf_qh)
#print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)
#print('ARI for MMBO using projection with L_{mix}: ', ARI_mbo_1_nor_Lf_Qh_1)
#print('purify for MMBO using projection with L_{mix}: ', purify_mbo_1_nor_Lf_Qh_1)
#print('inverse purify for MMBO using projection with L_{mix}: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
#print('NMI for MMBO using projection with L_{mix}: ', NMI_mbo_1_nor_Lf_Qh_1)


#    u_1_nor_Lf_Qh_individual_B,num_repeat_1_nor_Lf_Qh_B, MMBO_projection_modularity_list_B = mbo_modularity_1(num_nodes, num_communities, m, degree_B, u_init, 
#                                                D_mmbo_sym_B, V_mmbo_sym_B, tol, adj_mat_nparray)
#    u_1_nor_Lf_Qh_individual_label_B = vector_to_labels(u_1_nor_Lf_Qh_individual_B)
#    modularity_1_nor_lf_qh_B = skn.clustering.modularity(adj_mat_nparray,u_1_nor_Lf_Qh_individual_label_B,resolution=1)

#    modularity_MMBO_projection_sym_B_list.append(modularity_1_nor_lf_qh_B)
#    print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh_B)



# MMBO1 with inner step & sym normalized L_F & Q_H
#start_time_1_inner_nor_1 = time.time()
#    u_inner_nor_1,num_repeat_inner_nor, MMBO_inner_modularity_list = mbo_modularity_inner_step(num_nodes, num_communities, m,degree, dt_inner, u_init, 
#                                        D_mmbo_sym, V_mmbo_sym, tol, inner_step_count, adj_mat_nparray)
#time_MMBO_inner_step = time.time() - start_time_1_inner_nor_1
#print("MMBO using inner step with L_{mix}:-- %.3f seconds --" % ( time_eig_l_mix + time_initialize_u + time_MMBO_inner_step))
#    print('the number of MBO iteration for MMBO using inner step with L_{mix}: ',num_repeat_inner_nor)

#    u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)

#    MMBO_inner_cluster_list = np.unique(u_inner_nor_label_1)
#print('MMBO_projection_cluster list: ', MMBO_inner_cluster_list)
#    MMBO_inner_cluster = len(MMBO_inner_cluster_list)
#    print('the cluster MMBO using finite difference found: ', MMBO_inner_cluster)

#    modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_1,resolution=1)
#ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_membership)
#purify_mbo_1_inner_nor_1 = purity_score(gt_membership, u_inner_nor_label_1)
#inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_membership, u_inner_nor_label_1)
#NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_membership, u_inner_nor_label_1)

#    modularity_MMBO_inner_sym_list.append(modularity_1_inner_nor_1)
#    print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_1)
#print('ARI for MMBO using inner step with L_{mix}: ', ARI_mbo_1_inner_nor_1)
#print('purify for MMBO using inner step with L_{mix}: ', purify_mbo_1_inner_nor_1)
#print('inverse purify for MMBO using inner step with L_{mix}: ', inverse_purify_mbo_1_inner_nor_1)
#print('NMI for MMBO using inner step with L_{mix}: ', NMI_mbo_1_inner_nor_1)


#    u_inner_nor_B,num_repeat_inner_nor_B, MMBO_inner_modularity_list_B = mbo_modularity_inner_step(num_nodes, num_communities, m,degree_B, dt_inner, u_init, 
#                                        D_mmbo_sym_B, V_mmbo_sym_B, tol, inner_step_count, adj_mat_nparray)

#    u_inner_nor_label_B = vector_to_labels(u_inner_nor_B)
#    modularity_1_inner_nor_B = skn.clustering.modularity(adj_mat_nparray,u_inner_nor_label_B,resolution=1)
#ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, gt_membership)
#purify_mbo_1_inner_nor_1 = purity_score(gt_membership, u_inner_nor_label_1)
#inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(gt_membership, u_inner_nor_label_1)
#NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(gt_membership, u_inner_nor_label_1)

#    modularity_MMBO_inner_sym_B_list.append(modularity_1_inner_nor_B)
#    print('modularity for MMBO using inner step with L_{mix}: ', modularity_1_inner_nor_B)



numbers = []
for i in range(1, 26):
    numbers.append(i)


x_range = [1, 5, 10, 15, 20, 25]

# plot the spectra
#plt.plot(numbers, D_rw, ':',color='C2', label = "$L_{W_{rw}}$")
#plt.plot(numbers, D_sym, ':',color='C1', label = "$L_{W_{sym}}$")

#plt.plot(numbers, D_sym_signless_B_pos, ':',color='C9', label = "$L_{B_{sym}^+}$")
#plt.plot(numbers, D_sym_signless, ':',color='C7', label = "$Q_{P_{sym}}$")
#plt.plot(numbers, D_sym_signless_B_neg, ':',color='C8', label = "$Q_{B_{sym}^-}$")

#plt.plot(numbers, D_mmbo_rw, '--',color='C4', label = "$L_{W_{rw}} + Q_{P_{rw}}$")
#plt.plot(numbers, D_mmbo_sym, '--',color='C3', label = "$L_{W_{sym}} + Q_{P_{sym}}$")
#plt.plot(numbers, D_mmbo_rw_B, '-.',color='C6', label = "$L_{B_{rw}^+} + Q_{B_{rw}^-}$")
#plt.plot(numbers, D_mmbo_sym_B, '-.',color='C5', label = "$L_{B_{sym}^+} + Q_{B_{sym}^-}$")
#plt.title('Comparison of spectra.')
#plt.xticks(x_range)
#plt.xlabel('Index')
#plt.ylabel('Eigenvalue')
#plt.legend()
#plt.savefig('spectra_L.png')
#plt.show()



# plot number of iteration -- modularuty 
#plt.plot(numbers, HU_modularity_list, '-',color='C1', label = "Hu's method with $L_{W_{sym}}$")
#plt.plot(numbers, MMBO_projection_modularity_list, '--',color='C2', label = "MMBO using projection with $L_{W_{sym}},Q_{P_{sym}}$")
#plt.plot(numbers, MMBO_projection_modularity_list_B, '--',color='C3', label = "MMBO using projection with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
#plt.plot(numbers, MMBO_inner_modularity_list, ':',color='C4', label = "MMBO using finite difference with $L_{W_{sym}},Q_{P_{sym}}$")
#plt.plot(numbers, MMBO_inner_modularity_list_B, ':',color='C5', label = "MMBO using finite difference with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
#plt.title('Modularity Score with $m=10$')
#plt.xlabel('Number of iterations')
#plt.ylabel('Modularity')
#plt.legend()
#plt.savefig('Modularity_Score_SBM.png')
#plt.show()

# plot m -- modularity
#plt.plot(numbers, modularity_hu_sym_list, '-',color='C1', label = "Hu's method with $L_{W_{sym}}$")
#plt.plot(numbers, modularity_MMBO_projection_sym_list, '--',color='C2', label = "MMBO using projection with $L_{W_{sym}},Q_{P_{sym}}$")
#plt.plot(numbers, modularity_MMBO_projection_sym_B_list, '--',color='C3', label = "MMBO using projection with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
#plt.plot(numbers, modularity_MMBO_inner_sym_list, ':',color='C4', label = "MMBO using finite difference with $L_{W_{sym}},Q_{P_{sym}}$")
#plt.plot(numbers, modularity_MMBO_inner_sym_B_list, ':',color='C5', label = "MMBO using finite difference with $L_{B_{sym}^+},Q_{B_{sym}^-}$")
#plt.title('Modularity with the different choices of $m$.')
#plt.xlabel('Number of eigenvectors used')
#plt.ylabel('Modularity')
#plt.legend()
#plt.savefig('eig-Modularity_SBM.png')
#plt.show()


# Louvain
#start_time_louvain = time.time()
#G = nx.convert_matrix.from_scipy_sparse_matrix(W)
#G = nx.convert_matrix.from_numpy_array(adj_mat_nparray)
#partition_Louvain = community_louvain.best_partition(G, resolution=1)    # returns a dict
#louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
#louvain_array = np.asarray(louvain_list)
#print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))
#louvain_cluster = len(np.unique(louvain_array))
#print('number of clusters Louvain found: ',louvain_cluster)

#modularity_louvain = skn.clustering.modularity(adj_mat_nparray,louvain_array,resolution=1)
#ARI_louvain = adjusted_rand_score(louvain_array, gt_membership)
#purify_louvain = purity_score(gt_membership, louvain_array)
#inverse_purify_louvain = inverse_purity_score(gt_membership, louvain_array)
#NMI_louvain = normalized_mutual_info_score(gt_membership, louvain_array)

#print('modularity Louvain score: ', modularity_louvain)
#print('ARI Louvain  score: ', ARI_louvain)
#print('purify for Louvain : ', purify_louvain)
#print('inverse purify for Louvain : ', inverse_purify_louvain)
#print('NMI for Louvain  : ', NMI_louvain)


# Spectral clustering with k-means
#start_time_spectral_clustering = time.time()
#sc = SpectralClustering(n_clusters=num_communities, affinity='precomputed')
#assignment = sc.fit_predict(adj_mat_nparray)
#print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

#ass_vec = labels_to_vector(assignment)

#modularity_spectral_clustering = skn.clustering.modularity(adj_mat_nparray,assignment,resolution=1)
#ARI_spectral_clustering = adjusted_rand_score(assignment, gt_membership)
#purify_spectral_clustering = purity_score(gt_membership, assignment)
#inverse_purify_spectral_clustering = inverse_purity_score(gt_membership, assignment)
#NMI_spectral_clustering = normalized_mutual_info_score(gt_membership, assignment)

#print('modularity Spectral clustering score: ', modularity_spectral_clustering)
#print('ARI Spectral clustering  score: ', ARI_spectral_clustering)
#print('purify for Spectral clustering : ', purify_spectral_clustering)
#print('inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
#print('NMI for Spectral clustering: ', NMI_spectral_clustering)



# CNM algorithm (can setting resolution gamma)
#start_time_CNM = time.time()
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

#num_communities = 10
#iteration_proxy = num_communities
#MMBO_projection_cluster =5
#u_proxy = u_init.copy()
#u_1_nor_Lf_Qh_individual_1=[]

#while num_communities > MMBO_projection_cluster:
#    num_communities = iteration_proxy
#    u_1_nor_Lf_Qh_individual_1 = u_proxy
    
    #m = iteration_proxy
    
    #D_mmbo_1 = np.squeeze(eigenvalues_mmbo_sym[:m])
    #V_mmbo_1 = eigenvectors_mmbo_sym[:,:m]

#    u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,iteration_proxy, m, degree, u_proxy, 
#                                                D_mmbo, V_mmbo, tol, adj_mat_nparray)
    #time_MMBO_projection_sym = time.time() - start_time_1_nor_Lf_Qh_1
    #print("MMBO using projection with L_{mix}:-- %.3f seconds --" % (time_eig_l_mix + time_initialize_u + time_MMBO_projection_sym))
#    print('the number of MBO iteration for MMBO using projection with L_{mix}: ', num_repeat_1_nor_Lf_Qh_1)

#    u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)

#    MMBO_projection_cluster_list = np.unique(u_1_nor_Lf_Qh_individual_label_1)
#    print('MMBO_projection_cluster list: ', MMBO_projection_cluster_list)
#    MMBO_projection_cluster = len(MMBO_projection_cluster_list)
#    print('the cluster MMBO using projection found: ',MMBO_projection_cluster)

#    modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat_nparray,u_1_nor_Lf_Qh_individual_label_1,resolution=1)
#    ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, gt_membership)
#    purify_mbo_1_nor_Lf_Qh_1 = purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
#    inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)
#    NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(gt_membership, u_1_nor_Lf_Qh_individual_label_1)

#    print('modularity for MMBO using projection with L_{mix}: ', modularity_1_nor_lf_qh)
#    print('ARI for MMBO using projection with L_{mix}: ', ARI_mbo_1_nor_Lf_Qh_1)
#    print('purify for MMBO using projection with L_{mix}: ', purify_mbo_1_nor_Lf_Qh_1)
#    print('inverse purify for MMBO using projection with L_{mix}: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
#    print('NMI for MMBO using projection with L_{mix}: ', NMI_mbo_1_nor_Lf_Qh_1)
    
#    iteration_proxy = MMBO_projection_cluster
#    u_proxy = u_1_nor_Lf_Qh_individual_1

