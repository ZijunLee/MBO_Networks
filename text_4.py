import numpy as np
import scipy as sp
from MBO_Network import mbo_modularity_eig,data_generator
from sklearn.metrics.cluster import adjusted_rand_score
from graph_mbo.utils import spectral_clustering,vector_to_labels
import matplotlib.pyplot as plt


# Example 2 

# generate a random graph with community structure by the signed stochastic block model 
def SSBM_own(N, K):
    if N%K != 0:
        print("Wrong Input")

    else:
        s_matrix = -np.ones((N,N))
        cluster_size = N/K
        clusterlist = []
        for cs in range(K):
            clusterlist.append(int(cs*cluster_size))
        clusterlist.append(int(N))
        clusterlist.sort()
        #print(clusterlist)

        accmulate_size = []
        for quantity in range(len(clusterlist)-1):
            accmulate_size.append(clusterlist[quantity+1]-clusterlist[quantity])
        #print(accmulate_size)

        for interval in range(len(clusterlist)):
            for i in range(clusterlist[interval-1], clusterlist[interval]):
                for j in range(clusterlist[interval-1], clusterlist[interval]):
                    s_matrix[i][j] = 1
        #print(s_matrix)

        ground_truth = []
        for gt in range(len(accmulate_size)):
            ground_truth.extend([gt for y in range(accmulate_size[gt])])
        #print(ground_truth)

        ground_truth_v2 = []
        for gt in range(len(accmulate_size)):
            ground_truth_v2.extend([accmulate_size[gt] for y in range(accmulate_size[gt])])
        #print(ground_truth_v2)

    return s_matrix, ground_truth

N = 1500
K = 6
s_matrix, ground_truth = SSBM_own(N,K)
#print(s_matrix)
#print(ground_truth)


noise = 0.45
sparsity = 0.1
A_matrix = data_generator(s_matrix, noise, sparsity)
#print(s_matrix)

num_communities = K
m = K
dt = 0.1


V_output = mbo_modularity_eig(num_communities,m,dt,A_matrix, tol=1e-7, inner_step_count=3, 
                            normalized=False,symmetric=True, pseudospectral=True, signless=False)
#print(V_output)


V_label = vector_to_labels(V_output)
#print(V_label)
ARI = adjusted_rand_score(V_label,ground_truth)
print(ARI)

#noise = np.arange(0.05, 0.45, 0.04)
#print(noise)

#def compose_fct(data_generator, mbo_modularity_eig):
#    return lambda noise: mbo_modularity_eig(data_generator(noise))


#y_noise = compose_fct(mbo_modularity_eig,data_generator)
#print(y_noise(noise))

#def noise_mbo_to_ari(vector_to_labels,adjusted_rand_score,V_noise_variable):
#    noise_ari = adjusted_rand_score(vector_to_labels(V_noise_variable))
#    return noise_ari

#noise_label = compose_fct(vector_to_labels,compose_fct(mbo_modularity_eig,data_generator))
#noise_ari = adjusted_rand_score(noise_label,ground_truth)

#noise_variable = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
#ARI_noise = [1.0, 0.998, 0.99, 0.9761, 0.8629, 0.5962, 0.1433, 0.0041, 0.0002]
#plt.plot(noise_variable, ARI_noise)
#plt.title('N = 1500, K = 3, sparsity = 0.02')
#plt.xlabel('noise')
#plt.ylabel('ARI')
#plt.show()