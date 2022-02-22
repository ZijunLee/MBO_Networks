from pickle import FALSE
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from MBO_signednetwork import mbo_modularity
from MBO_Network import mbo_modularity_1, mbo_modularity_1_normalized_lf, mbo_modularity_2, mbo_modularity_inner_step,adj_to_laplacian_signless_laplacian
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
import time
#from networkx.algorithms.community.centrality import girvan_newman
from Network_function import greedy_modularity_communities

start_time = time.time()

# Parameter setting

num_communities = 3
m = 1 * num_communities
#m_1 = 2 * num_communities
#m = 3
dt = 0.5
tol = 0
inner_step_count = 3

eta_1 = 1
eta_06 = 0.6
eta_05 = 0.5
eta_03 = 1.3
inner_step_count =3


# Example 1: comparsion MBO Schene and spectral_clustering in Zachary's Karate Club graph
G = nx.karate_club_graph()
#print(type(G))
gt_membership = [G.nodes[v]['club'] for v in G.nodes()]
#print(gt_membership)

gt_number = []
for i in gt_membership:
    if i == "Mr. Hi":
        gt_number.append(1)
    elif i =="Officer":
        gt_number.append(0)    
#print('gt_number: ',type(gt_number))   # gt_number is a list
#gt_vec = labels_to_vector(gt_number)
#print(gt_vec)

gt_label_dict = []
len_gt_label = []

for e in range(len(gt_number)):
    len_gt_label.append(e)

gt_label_dict = dict(zip(len_gt_label, gt_number))
#print(gt_label_dict)


adj_mat = nx.convert_matrix.to_numpy_matrix(G)
#print('adj matrix: ',adj_mat)

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1, signless_laplacian_null_model_1, nor_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)
num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, signless_laplacian_null_model_06, nor_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_06,target_size=None)
num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, signless_laplacian_null_model_05, nor_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_05,target_size=None)



## Test MMBO 1 with unnormalized L_F
tol = 0
gamma_1 = 1
#u_1,num_repeat_1 = mbo_modularity_1(num_communities,m,adj_mat, tol,gamma_1)
#print(num_repeat_1)
#print(u_1)

#u_1_label = vector_to_labels(u_1)
#u_1_label_dict = label_to_dict(u_1_label)
#print(u_1_label_dict)


## Test MMBO 1 with normalized L_F
tol = 0
gamma_1 = 1
u_1_nor_individual,num_repeat_1_nor = mbo_modularity_1_normalized_lf(num_nodes_1,num_communities, m_1,degree_1, nor_graph_laplacian_1,signless_laplacian_null_model_1, 
                                                tol, target_size_1,eta_1, eps=1)
#print(num_repeat_1)
#print(u_1)

u_1_label_nor_1 = vector_to_labels(u_1_nor_individual)
u_1_label_dict_nor_1 = label_to_dict(u_1_label_nor_1)
#print(u_1_label_dict)

## Test MMBO 2

#u_2,nun_times_2 = mbo_modularity_2(num_communities,m,adj_mat, tol=0, gamma=1)
#print(nun_times_2)
#u_2_label = vector_to_labels(u_2)
#u_2_label_dict = label_to_dict(u_2_label)


## Test MMBO with inner step Nt
tol = 0
gamma_inner = 1
inner_step_count=3
dt_inner = 0.1

#u_inner,num_repeat_inner = mbo_modularity_inner_step(num_communities,m,adj_mat, dt_inner, tol,inner_step_count, gamma_inner)
#print(num_repeat_1)
#print(u_1)

#u_inner_label = vector_to_labels(u_inner)
#u_inner_label_dict = label_to_dict(u_inner_label)
#print(u_1_label_dict)


## Test original MBO
dt_ori = 0.3
#u_original = mbo_modularity(num_communities,m,dt_ori,adj_mat, tol=1e-7, inner_step_count=3,
#                            pseudospectral=False,symmetric=True,signless=False,modularity=False)

#u_ori_label = vector_to_labels(u_original)
#u_ori_label_dict = label_to_dict(u_ori_label)


#modularity_1 = get_modularity_original(adj_mat,u_1)
#print('modularity 1: ',modularity_1)

#mod_ver2 = get_modularity(G,u_2_label_dict)
#print('mod version 2: ',mod_ver2)


# Louvain algorithm (can setting resolution gamma)
#partition_Louvain = co.best_partition(G, resolution=0.5)
#print('Louvain:',partition_Louvain)


# CNM algorithm (can setting resolution gamma)
#partition_CNM = nx_comm.greedy_modularity_communities(G)

#partition_CNM_list = [list(x) for x in partition_CNM]

#partition_CNM_expand = sum(partition_CNM_list, [])

#num_cluster_CNM = []
#for cluster in range(len(partition_CNM_list)):
#    for number_CNM in range(len(partition_CNM_list[cluster])):
#        num_cluster_CNM.append(cluster)

#print(partition_CNM_list)
#CNM_dict = dict(zip(partition_CNM_expand, num_cluster_CNM))
#print('CNM: ',CNM_dict)


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


# Spectral clustering with k-means
num_communities = 2
sc = SpectralClustering(n_clusters=2, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)

#degree = np.array(np.sum(adj_mat, axis=1)).flatten()
#num_nodes = len(degree)
#graph_laplacian, degree = sp.sparse.csgraph.laplacian(adj_mat, return_diag=True)
#degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
#graph_laplacian = degree_diag - adj_mat
#degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
#sym_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)
#D, V = eigsh(
#    sym_graph_laplacian,
#    k=num_communities,
#    v0=np.ones((sym_graph_laplacian.shape[0], 1)),
#    which="SA",)
#V = V[:, 1:].reshape((-1, 1))
#kmeans = KMeans(n_clusters=2)
#kmeans.fit(V)
#assignment = kmeans.predict(V)
#print('spectral clustering: ',len(assignment))
ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)



## Compute modularity scores

#modu_gt = co.modularity(gt_label_dict,G)
#modu_1 = co.modularity(u_1_label_dict,G)
modu_1_nor_Lf_1 = co.modularity(u_1_label_dict_nor_1,G)
#modu_2 = co.modularity(u_2_label_dict,G)
#modu_inner = co.modularity(u_inner_label_dict,G)
#modu_orig = co.modularity(u_ori_label_dict,G)
#modu_louvain = co.modularity(partition_Louvain,G)
#modu_CNM = co.modularity(CNM_dict,G)
modu_GN = co.modularity(GN_dict,G)
#modu_sc = co.modularity(ass_dict,G)
#modularity_GN_1 = get_modularity(G,GN_dict)
#modularity_CNM_2 = nx_comm.modularity(G,partition_CNM_list)
#modu_louvain = nx_comm.modularity(G, partition_Louvain)



#print('modularity_gt score:',modu_gt)
#print('modularity_1 unnormalized score:',modu_1)
print('modularity_1 normalized score:',modu_1_nor_Lf_1)
#print('modularity_2 score:',modu_2)
#print('modularity_inner_step score:',modu_inner)
#print('modularity_original score:',modu_orig)
#print('modularity_Louvain score:',modu_louvain)
#print('modularity_CNM score:',modu_CNM)
print('modularity_GN score:',modu_GN)
#print('modularity_GN_1 score:',modularity_GN_1)
#print('modularity_CNM_2 score:',modularity_CNM_2)
#print('modularity_spectral clustering score:',modu_sc)




## Compare ARI 
#ARI_mbo_1 = adjusted_rand_score(u_1_label, gt_number)
#ARI_mbo_1_nor = adjusted_rand_score(u_1_label_nor, gt_number)
#ARI_mbo_2 = adjusted_rand_score(u_2_label, gt_number)
#ARI_mbo_ori = adjusted_rand_score(u_ori_label, gt_number)
#ARI_mbo_inner = adjusted_rand_score(u_inner_label, gt_number)
ARI_spectral_clustering = adjusted_rand_score(assignment, gt_number)
#ARI_gn = adjusted_rand_score(partition_GN, gt_number)



#print('ARI for MBO_1 unnormalized: ', ARI_mbo_1)
#print('ARI for MBO_1 normalized: ', ARI_mbo_1_nor)
#print('ARI for MBO_2: ', ARI_mbo_2)
#print('ARI for MBO_inner_step: ', ARI_mbo_inner)
#print('ARI for MBO_original: ', ARI_mbo_ori)
print('ARI for spectral clustering: ', ARI_spectral_clustering)
#print('ARI for MBO_3_negative: ', ARI_mbo_3_negative)
#print('ARI for MBO_3_positive: ', ARI_mbo_3_positive)
#print('ARI for GN: ', ARI_gn)

#print("--- %.3f seconds ---" % (time.time() - start_time))

#colors1 = ["#FF0000", "#0000FF"]
#colors2 = ["#458B74", "#FF9912"]
colors3 = ["#FF0000", "#FF9912"]
plt.figure()
fig, axes = plt.subplots(4,1,figsize=(13, 37))
axs = axes.flatten()
# loc = nx.kamada_kawai_layout(G)
loc = nx.spring_layout(G)

#for image in range(4):
#    nx.draw(G, with_labels = True, node_size=1000, edge_color="black", pos=loc, ax = axes[image])
#    plt.title('MMBO1 with $L_f$111')
#    #nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color="grey", pos=loc, ax = axes[image])
    
#    for p in range(2):
#        nodes = np.argwhere(u_original[:, p])
#        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[p], pos=loc, ax = axes[1])
#    plt.title('MMBO1 with $L_f$222')
    
#    for i in range(2):
#        nodes = np.argwhere(u_1[:, i])
#        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[i], pos=loc, ax = axes[2])
#    plt.title('MMBO1 with $L_f$333')
    
#    for q in range(2):
#        nodes = np.argwhere(u_1_nor[:, q])
#        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[q], pos=loc, ax = axes[3])
#    plt.title('MMBO1 with $L_f$444')
    #    for k in range(2):
    #        nodes = np.argwhere(u_2[:, k])
    #        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[k], pos=loc, ax = axes[4])
    #nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color="grey", pos=loc)
    #for j in range(2):
    #    nodes = np.argwhere(ass_vec[:,j])
    #    nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors3[j], pos=loc,ax = axes[3])

#plt.savefig('Figure_3.png')
#plt.show()


