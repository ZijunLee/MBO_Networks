from turtle import color
import matplotlib.pyplot as plt
from matplotlib import lines
import math
import numpy as np
import community as co
import networkx as nx
import networkx.algorithms.community as nx_comm
from numpy import number
#from MBO_Network import  mbo_modularity_1_normalized_lf,mbo_modularity_2
#from MBO_Network import mbo_modularity_inner_step,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh
from graph_mbo.utils import get_fidelity_term,vector_to_labels,label_to_dict
import time
from MBO_signednetwork import mbo_modularity
from MBO_Network import mbo_modularity_1,mbo_modularity_1_normalized_lf,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh,mbo_modularity_2,adj_to_laplacian_signless_laplacian


num_communities = 8
#num_communities = [2, 3, 4, 5, 6, 7, 8]
m = 1 * num_communities
eta_1 =1
tol = 0

G = nx.karate_club_graph()
adj_mat = nx.convert_matrix.to_numpy_matrix(G)

modularity_1_unnor_list = []

num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1, signless_laplacian_null_model_1, nor_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,num_communities,m,eta_1,target_size=None)
#num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, signless_laplacian_null_model_06, nor_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(adj_mat,i,m,eta_06,target_size=None)
#num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, signless_laplacian_null_model_05, nor_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(adj_mat,i,m,eta_05,target_size=None)

accumulator = 0

for _ in range(30):
    # mmbo 1 with unnormalized L_F and gamma=1
    u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,num_communities, m,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                    tol, target_size_1,eta_1, eps=1)   
    u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
    u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)
    #print('u_1_unnormalized label: ', u_1_unnor_individual_label)

    modularity_1_unnor_individual = co.modularity(u_1_unnor_individual_label_dict,G)
    #modularity_1_unnor_list.append(modularity_1_unnor_individual)

    # reading = modularity_1_unnor_individual()
    accumulator += modularity_1_unnor_individual
average = accumulator / 30
#print(average)

accumulator_2_nor_individual_1 =0
average_2_nor_1_list =[]
for _ in range(30):
# mmbo 2 with normalized & gamma = 1
    u_2_individual, num_repeat_2 = mbo_modularity_2(num_communities, m, adj_mat,null_model_eta_1, tol,eta_1,eps=1,
                    target_size=None, fidelity_type="karate", initial_state_type="random") 
    u_2_individual_label = vector_to_labels(u_2_individual)
    u_2_individual_label_dict = label_to_dict(u_2_individual_label)
    
    modularity_2_individual = co.modularity(u_2_individual_label_dict,G)
    #modularity_2_list.append(modularity_2_individual)

    accumulator_2_nor_individual_1 += modularity_2_individual
average_2_nor_1 = accumulator_2_nor_individual_1 / 30
#average_2_nor_1_list.append(average_2_nor_1)
print(average_2_nor_1)