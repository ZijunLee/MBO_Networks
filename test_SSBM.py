from pickle import FALSE
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from MBO_signednetwork import mbo_modularity
from MBO_Network import mbo_modularity_1, mbo_modularity_2,data_generator,SSBM_own
from graph_mbo.utils import spectral_clustering,vector_to_labels, get_modularity_original,get_modularity,labels_to_vector,label_to_dict
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities,girvan_newman
import numpy as np
import scipy as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh,eigs
import community as co
import time

# MMBO 3

gamma_positive=1
gamma_negative=1
N = 12
K = 3
sparsity = 0.2
num_communities = K
m = K
tol=1e-7
noise =0.1


#noise_variable = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
#noise_variable_list = []
#stride = 0.05
#number_quantity = 9
#for nummer in range(1, number_quantity+1):
#    noise_variable_list.append(round(nummer*stride,2))
#print(noise_variable_list)

#ARI_list = []
#for i in noise_variable_list:
#    N = 120
#    K = 2
#    s_matrix, ground_truth = SSBM_own(N,K)
#    sparsity = 0.02

#    A_matrix_individual = data_generator(s_matrix, i, sparsity)
#    num_communities = K
#    m = K
#    tol=1e-7
#    gamma =1
#    dt = 0.1

#    u_3_positive,u_3_negative = mbo_modularity_3(num_communities,m,A_matrix_individual, tol, gamma_positive,gamma_negative)
#    V_label_individual = vector_to_labels(V_output_individual)
#    modularity = get_modularity_original(A_matrix_individual,V_output_individual)
#    print('modularity score:',modularity)

#    ARI_individual = adjusted_rand_score(V_label_individual,ground_truth)
#    print(ARI_individual)
#    ARI_list.append(ARI_individual)

#ARI_noise = [1.0, 0.998, 0.99, 0.9761, 0.8629, 0.5962, 0.1433, 0.0041, 0.0002]
#plt.plot(noise_variable, ARI_list)
#plt.title('N = 1200, K = 5, sparsity = 0.02')
#plt.xlabel('noise')
#plt.ylabel('ARI')
#plt.show()


s_matrix, ground_truth = SSBM_own(N,K)
print('ground truth: ',ground_truth)
A_matrix_individual = data_generator(s_matrix, noise, sparsity)
print('A_matrix, ', A_matrix_individual)

#u_2,nun_times_2 = mbo_modularity_2(num_communities,m,A_matrix_individual, tol=1e-7, gamma=1)
#print(u_2)
#u_2_label = vector_to_labels(u_2)

#ARI_u_2 = adjusted_rand_score(u_2_label,ground_truth)

#print(ARI_u_2)

#u_3_positive,u_3_negative = mbo_modularity_3(num_communities,m,A_matrix_individual, tol, gamma_positive,gamma_negative)
#u_3_mix = u_3_positive - u_3_negative

#print('u_3 positive: ',u_3_positive)
#print('u_3 negative: ',u_3_negative)
#print('u_3_mix: ', u_3_mix)