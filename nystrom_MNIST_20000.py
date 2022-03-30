import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sknetwork as skn
plt.style.use('ggplot')
import os,sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.sparse import csc_matrix
from numpy.linalg import multi_dot
from numpy.linalg import norm
from sklearn.cluster import KMeans, SpectralClustering
from math import pi
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score,f1_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
from sklearn.svm import LinearSVC
import time
from graph_cut.data.read_mnist import Read_mnist, subsample
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new
from MBO_Network import construct_null_model, mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original
from graph_mbo.utils import purity_score,inverse_purity_score
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from community import community_louvain
import csv


def nystrom_new(train_data, gamma, num_nystrom=500, k=200, seed=44):

    rng = np.random.RandomState(seed)
    n_samples = train_data.shape[0]
    idx = rng.choice(n_samples, num_nystrom)

    train_data_idx = train_data[idx, :]

    W = rbf_kernel(train_data_idx, train_data_idx, gamma=gamma)

    u, s, vt = linalg.svd(W, full_matrices=False)
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k, :]

    M = np.dot(u, np.diag(1/np.sqrt(s)))

    C_training = rbf_kernel(train_data, train_data_idx, gamma=gamma) 

    X_new_training = np.dot(C_training, M)

    return X_new_training

## parameter setting
dt_inner = 1
num_communities = 10
m = 1 * num_communities
dt = 0.5
tol = 1e-7
inner_step_count =3
eta_1 =1

gpath = '/'.join(os.getcwd().split('/')[:-1])

#raw_data, labels = Read_mnist(digits = [4,9],path = gpath+'/MBO_signed_graphs/graph_cut/data') 
#raw_data = raw_data/255.
full_data, full_labels = Read_mnist(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
#full_data, full_labels = Read_mnist(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
#full_data, full_labels = Read_mnist(digits = range(10))
full_data = full_data/255.

sample_data,sample_labels = subsample(sample_num = 2000, rd = full_data, labels = full_labels)

pca = PCA(n_components = 50, svd_solver='full')
pca.fit_transform(full_data)
train_data = pca.transform(sample_data)


n1, p = train_data.shape

print("Training samples :", n1)
print("Features:", p)

gamma = 1. / p

Z_training = nystrom_new(train_data, gamma, num_nystrom=500, k=50, seed=44)

# builf adjacency matrix W
adj_mat = build_affinity_matrix_new(Z_training,gamma=1, affinity='rbf',n_neighbors=10, neighbor_type='knearest')
print('adj_matrix shape: ', adj_mat.shape)

# construct null model based on W
null_model = construct_null_model(adj_mat)

# compute symmetric L_F, random walk L_F, symmetric Q_H, random walk Q_H
num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

# construct L_{mix}
start_time_lap_mix = time.time()
l_mix =  sym_graph_lap + sym_signless_lap
time_l_mix = time.time() - start_time_lap_mix
print("compute laplacian_mix:-- %.3f seconds --" % (time_l_mix))

start_time_1_nor_Lf_Qh_1 = time.time()
## Test MMBO 1 with normalized L_F
u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m_1, dt, l_mix,
                                                 tol, target_size, eta_1)

time_u_1_nor_Lf_Qh = time.time() - start_time_1_nor_Lf_Qh_1           
print("MMBO with normalized L_F & Q_H (K=10, m=K):-- %.3f seconds --" % (time_u_1_nor_Lf_Qh))
print('u_1 nor L_F & Q_H number of iteration(K=10 and m=K): ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)

modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, sample_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)

print(' modularity normalized L_F & Q_H score(K=10 and m=K): ', modularity_1_nor_lf_qh)
print(' ARI normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print(' purify for MMBO1 normalized L_F & Q_H: ', purify_mbo_1_nor_Lf_Qh_1)
print(' inverse purify for MMBO1 normalized L_F & Q_H: ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print(' NMI for MMBO1 normalized L_F & Q_H: ', NMI_mbo_1_nor_Lf_Qh_1)



start_time_1_inner_nor_1 = time.time()

# MMBO1 with inner step & sym normalized L_F and gamma=1
u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes, num_communities, m_1, l_mix, dt_inner,
                                                         tol, target_size, inner_step_count)
u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
time_u_inner_nor = time.time() - start_time_1_inner_nor_1
print("MMBO1 with inner step & symmetric normalized L_F & Q_H, and gamma=1:-- %.3f seconds --" % (time_u_inner_nor))
print('inner step & symmetric normalized L_F & Q_H, number of iteration(K=10 and m=K): ', num_repeat_inner_nor)

modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat,u_inner_nor_label_1,resolution=0.5)
ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, sample_labels)
purify_mbo_1_inner_nor_1 = purity_score(sample_labels, u_inner_nor_label_1)
inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(sample_labels, u_inner_nor_label_1)
NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(sample_labels, u_inner_nor_label_1)

print(' modularity inner step sym normalized score: ', modularity_1_inner_nor_1)
print(' ARI inner step sym normalized score: ', ARI_mbo_1_inner_nor_1)
print(' purify for MMBO1 inner step with sym normalized: ', purify_mbo_1_inner_nor_1)
print(' inverse purify for MMBO1 inner step with sym normalized : ', inverse_purify_mbo_1_inner_nor_1)
print(' NMI for MMBO1 inner step with sym normalized: ', NMI_mbo_1_inner_nor_1)



start_time_hu_original = time.time()

# test HU original MBO
u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m_1, degree, dt_inner,
                                              sym_graph_lap, tol, target_size ,inner_step_count) 
u_hu_label_1 = vector_to_labels(u_hu_vector)
#u_hu_dict_1 = label_to_dict(u_hu_label_1)
time_HU = time.time() - start_time_hu_original
print("HU original MBO:-- %.3f seconds --" % (time_HU))
print('number of iteration for HU MBO: ', num_iter_HU)

modu_hu_original_1 = skn.clustering.modularity(adj_mat,u_hu_label_1,resolution=0.5)
ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, sample_labels)
purify_hu_original_1 = purity_score(sample_labels, u_hu_label_1)
inverse_purify_hu_original_1 = inverse_purity_score(sample_labels, u_hu_label_1)
NMI_hu_original_1 = normalized_mutual_info_score(sample_labels, u_hu_label_1)

print(' modularity score for HU original MBO: ', modu_hu_original_1)
print(' ARI for HU original MBO: ', ARI_hu_original_1)
print(' purify for HU original MBO : ', purify_hu_original_1)
print(' inverse purify for HU original MBO : ', inverse_purify_hu_original_1)
print(' NMI for HU original MBO : ', NMI_hu_original_1)



start_time_louvain = time.time()
G = nx.convert_matrix.from_numpy_array(adj_mat)

partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)
time_louvain = time.time() - start_time_louvain
print("Louvain:-- %.3f seconds --" % (time_louvain))

modularity_louvain = skn.clustering.modularity(adj_mat,louvain_array,resolution=0.5)
ARI_louvain = adjusted_rand_score(louvain_array, sample_labels)
purify_louvain = purity_score(sample_labels, louvain_array)
inverse_purify_louvain = inverse_purity_score(sample_labels, louvain_array)
NMI_louvain = normalized_mutual_info_score(sample_labels, louvain_array)

print(' modularity Louvain score: ', modularity_louvain)
print(' ARI Louvain score: ', ARI_louvain)
print(' purify for Louvain : ', purify_louvain)
print(' inverse purify for Louvain : ', inverse_purify_louvain)
print(' NMI for Louvain: ', NMI_louvain)



start_time_spectral_clustering = time.time()

# Spectral clustering with k-means
sc = SpectralClustering(n_clusters=10, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)

ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)

time_spectral_clustering = time.time() - start_time_spectral_clustering
print("spectral clustering algorithm:-- %.3f seconds --" % (time_spectral_clustering))

modularity_spectral_clustering = skn.clustering.modularity(adj_mat,assignment,resolution=0.5)
ARI_spectral_clustering = adjusted_rand_score(assignment, sample_labels)
purify_spectral_clustering = purity_score(sample_labels, assignment)
inverse_purify_spectral_clustering = inverse_purity_score(sample_labels, assignment)
NMI_spectral_clustering = normalized_mutual_info_score(sample_labels, assignment)

print(' modularity Spectral clustering score(K=10 and m=K): ', modularity_spectral_clustering)
print(' ARI Spectral clustering  score: ', ARI_spectral_clustering)
print(' purify for Spectral clustering : ', purify_spectral_clustering)
print(' inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
print(' NMI for Spectral clustering: ', NMI_spectral_clustering)


# store results as a .csv
testarray_MNIST = ["compute laplacian_mix", "MMBO with normalized L_F & Q_H (K=10, m=K) time"
             "u_1 nor L_F & Q_H number of iteration", "modularity normalized L_F & Q_H score",
             "ARI normalized L_F & Q_H", "purify for MMBO1 normalized L_F & Q_H",
             "inverse purify for MMBO1 normalized L_F & Q_H", "NMI for MMBO1 normalized L_F & Q_H",
             "MMBO1 with inner step & symmetric normalized L_F & Q_H time", "inner step & symmetric normalized L_F & Q_H, number of iteration",
             "modularity inner step sym normalized", "ARI inner step sym normalized",
             "purify for MMBO1 inner step with sym normalized", "inverse purify for MMBO1 inner step with sym normalized",
             "NMI for MMBO1 inner step with sym normalized", "HU original MBO running time",
             "number of iteration for HU MBO", "modularity score for HU",
             "ARI for HU","purify for HU",
             "inverse purify for HU", "NMI for HU",
             "Louvain running time", "modularity Louvain",
             "ARI Louvain", "purify for Louvain",
             "inverse purify for Louvain", "NMI for Louvain",
             "spectral clustering running time", "modularity Spectral clustering",
             "ARI Spectral clustering", "purify for Spectral clustering",
             "inverse purify for Spectral clustering", "NMI for Spectral clustering"]


resultarray_MNIST = [time_l_mix, time_u_1_nor_Lf_Qh,
               num_repeat_1_nor_Lf_Qh_1, modularity_1_nor_lf_qh,
               ARI_mbo_1_nor_Lf_Qh_1, purify_mbo_1_nor_Lf_Qh_1,
               inverse_purify_mbo_1_nor_Lf_Qh_1, NMI_mbo_1_nor_Lf_Qh_1,
               time_u_inner_nor, num_repeat_inner_nor,
               modularity_1_inner_nor_1, ARI_mbo_1_inner_nor_1,
               purify_mbo_1_inner_nor_1, inverse_purify_mbo_1_inner_nor_1,
               NMI_mbo_1_inner_nor_1, time_HU,
               num_iter_HU, modu_hu_original_1,
               ARI_hu_original_1, purify_hu_original_1, 
               inverse_purify_hu_original_1, NMI_hu_original_1,
               time_louvain, modularity_louvain,
               ARI_louvain, purify_louvain,
               inverse_purify_louvain, NMI_louvain,
               time_spectral_clustering, modularity_spectral_clustering,
               ARI_spectral_clustering, purify_spectral_clustering,
               inverse_purify_spectral_clustering, NMI_spectral_clustering ]

with open('nystrom_MNIST_20000.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(testarray_MNIST)
    wr.writerow(resultarray_MNIST)