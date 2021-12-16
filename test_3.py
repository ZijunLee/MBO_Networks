from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from MBO_Network import mbo_modularity_eig,SSBM_own,data_generator
from graph_mbo.utils import spectral_clustering,vector_to_labels
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
from sklearn.metrics import adjusted_rand_score




# Parameter setting

num_communities = 2
m = 2
dt = 0.15
tol = 10**(-7)
inner_step_count = 10


# Example 1: comparsion MBO Schene and spectral_clustering in Zachary's Karate Club graph
G = nx.karate_club_graph()
gt_membership = [G.nodes[v]['club'] for v in G.nodes()]
#print(gt_membership)

gt_number = []
for i in gt_membership:
    if i == "Mr. Hi":
        gt_number.append(1)
    elif i =="Officer":
        gt_number.append(0)    
#print(gt_number)

adj_mat = nx.convert_matrix.to_numpy_matrix(G)
u = mbo_modularity_eig(num_communities,m,dt,adj_mat, tol=1e-7, inner_step_count=100, 
                            pseudospectral=False, symmetric=True, signless=False)
#print(u)

u_label = vector_to_labels(u)

v = spectral_clustering(adj_mat,2)
#print(v)
v_label = vector_to_labels(v)
#print(v_label)

ARI_mbo = adjusted_rand_score(u_label, gt_number)
ARI_spectral_clustering = adjusted_rand_score(v_label, gt_number)


print('ARI for MBO: ', ARI_mbo)
print('ARI for spectral clustering: ', ARI_spectral_clustering)

colors1 = ["#FF0000", "#0000FF"]
colors2 = ["#458B74", "#FF9912"]
plt.figure()
fig, axes = plt.subplots(3,1,figsize=(13, 21))
axs = axes.flatten()
# loc = nx.kamada_kawai_layout(G)
loc = nx.spring_layout(G)

for image in range(3):
    nx.draw(G, with_labels = True, node_size=1000, edge_color="black", pos=loc, ax = axes[image])
    plt.xlabel('111')
    #nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color="grey", pos=loc, ax = axes[image])
    for i in range(2):
        nodes = np.argwhere(u[:, i])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors1[i], pos=loc, ax = axes[1])
        plt.xlabel('222')
    #nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color="grey", pos=loc)
    for j in range(2):
        nodes = np.argwhere(v[:, i])
        nx.draw_networkx_nodes(G, node_size=1000, nodelist=nodes.flatten(), node_color=colors2[j], pos=loc,ax = axes[2])

plt.savefig('Figure_3.png')
plt.show()


