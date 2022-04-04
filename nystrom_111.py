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
from graph_cut.data.read_mnist import Read_mnist_function, subsample
from graph_cut_util import LaplacianClustering,build_affinity_matrix_new
from MBO_Network import mbo_modularity_1, adj_to_laplacian_signless_laplacian, mbo_modularity_inner_step, mbo_modularity_hu_original,construct_null_model
from graph_mbo.utils import purity_score,inverse_purity_score
from graph_mbo.utils import vector_to_labels, labels_to_vector,label_to_dict, purity_score,inverse_purity_score, dict_to_list_set
from community import community_louvain


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
#full_data, full_labels = Read_mnist(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
full_data, full_labels = Read_mnist_function(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
#full_data, full_labels = Read_mnist(digits = range(10))
full_data = full_data/255.

sample_data,sample_labels = subsample(sample_num = 1500, rd = full_data, labels = full_labels)

pca = PCA(n_components = 50, svd_solver='full')
pca.fit_transform(full_data)
#train_data = pca.transform(sample_data)
train_data = pca.transform(sample_data)

#X, y = make_classification(n_samples=100000)
#X_train, X_test, y_train, y_test = train_test_split(full_data, full_labels,test_size=0.25, random_state=42)

#print('y_train shape: ',y_train.shape)
#print('y_test shape: ',y_test.shape)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

#pca = PCA(n_components = 50, svd_solver='full')
#X_train = pca.fit_transform(X_train)
#train_data = pca.transform(sample_data)
#X_test = pca.transform(X_test)

n1, p = train_data.shape
#n2 = X_test.shape[0]

print("Training samples :", n1)
#print("Test samples:", n2)
print("Features:", p)

def nystrom(X_train, X_test, gamma, c=500, k=200, seed=44):

    rng = np.random.RandomState(seed)
    n_samples = X_train.shape[0]
    idx = rng.choice(n_samples, c)

    X_train_idx = X_train[idx, :]
    W = rbf_kernel(X_train_idx, X_train_idx, gamma=gamma)

    u, s, vt = linalg.svd(W, full_matrices=False)
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k, :]

    M = np.dot(u, np.diag(1/np.sqrt(s)))

    C_train = rbf_kernel(X_train, X_train_idx, gamma=gamma)
    C_test = rbf_kernel(X_test, X_train_idx, gamma=gamma)

    X_new_train = np.dot(C_train, M)
    X_new_test = np.dot(C_test, M)

    return X_new_train, X_new_test


def nystrom_new(train_data, gamma, c=500, k=200, seed=44):

    rng = np.random.RandomState(seed)
    #n_samples = X_train.shape[0]
    n_samples = train_data.shape[0]
    idx = rng.choice(n_samples, c)

    #X_train_idx = X_train[idx, :]
    train_data_idx = train_data[idx, :]
    #W = rbf_kernel(X_train_idx, X_train_idx, gamma=gamma)
    W = rbf_kernel(train_data_idx, train_data_idx, gamma=gamma)

    u, s, vt = linalg.svd(W, full_matrices=False)
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k, :]

    M = np.dot(u, np.diag(1/np.sqrt(s)))

    #C_train = rbf_kernel(X_train, X_train_idx, gamma=gamma)
    #C_test = rbf_kernel(X_test, X_train_idx, gamma=gamma)
    C_training = rbf_kernel(train_data, train_data_idx, gamma=gamma) 

    #X_new_train = np.dot(C_train, M)
    #X_new_test = np.dot(C_test, M)
    X_new_training = np.dot(C_training, M)

    #return X_new_train, X_new_test
    return X_new_training


gamma = 1. / p
#Z_train, Z_test = nystrom(X_train, X_test, gamma, c=500, k=50, seed=44)
#print('Z_train shape: ', Z_train.shape)
#print('Z_test shape: ', Z_test.shape)

Z_training = nystrom_new(train_data, gamma, c=500, k=50, seed=44)

#t0 = time.time()
#clf = LinearSVC(dual=False)
#clf.fit(Z_train, y_train)
#clf.fit(Z_training,sample_labels)
#print("done in %0.3fs" % (time.time() - t0))


#t1 = time.time()
#accuracy = clf.score(Z_test, y_test)
#accuracy = clf.score(Z_training, sample_labels)
#print("done in %0.3fs" % (time.time() - t1))
#print("classification accuracy: %0.3f" % accuracy)

#t0_x = time.time()
#clf = LinearSVC(dual=False)
#clf.fit(X_train, y_train)
#print("done in %0.3fs" % (time.time() - t0_x))

#t1_x = time.time()
#accuracy = clf.score(Z_test, y_test)
#accuracy_x = clf.score(X_test, y_test)
#print("done in %0.3fs" % (time.time() - t1_x))
#print("classification accuracy: %0.3f" % accuracy_x)

adj_mat = build_affinity_matrix_new(Z_training,gamma=gamma, affinity='rbf',n_neighbors=10, neighbor_type='knearest')
print('adj_matrix shape: ', adj_mat.shape)

null_model = construct_null_model(adj_mat)

num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

start_time_1_nor_Lf_Qh_1 = time.time()
## Test MMBO 1 with normalized L_F

#u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_given_eig(num_communities, eigenvalues, eigenvectors, 
#                                                            degree_1, dt, tol, inner_step_count)

u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1(num_nodes,num_communities, m_1, dt, sym_graph_lap, sym_signless_lap,
                                                 tol, target_size, eta_1)
                                                
print("MMBO1 with normalized L_F & Q_H (K=10, m=K):-- %.3f seconds --" % (time.time() - start_time_1_nor_Lf_Qh_1))
print('u_1 nor L_F & Q_H number of iteration(K=10 and m=K): ', num_repeat_1_nor_Lf_Qh_1)
u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
#u_1_nor_individual_label_dict_1 = label_to_dict(u_1_nor_individual_label_1)
#print('u_1_nor_Lf_Qh_individual_label_1: ', u_1_nor_Lf_Qh_individual_label_1.shape)

modularity_1_nor_lf_qh = skn.clustering.modularity(adj_mat,u_1_nor_Lf_Qh_individual_label_1,resolution=0.5)
ARI_mbo_1_nor_Lf_Qh_1 = adjusted_rand_score(u_1_nor_Lf_Qh_individual_label_1, sample_labels)
purify_mbo_1_nor_Lf_Qh_1 = purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
inverse_purify_mbo_1_nor_Lf_Qh_1 = inverse_purity_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)
NMI_mbo_1_nor_Lf_Qh_1 = normalized_mutual_info_score(sample_labels, u_1_nor_Lf_Qh_individual_label_1)

print(' modularity_1 normalized L_F & Q_H score(K=10 and m=K): ', modularity_1_nor_lf_qh)
print('average ARI_1 normalized L_F & Q_H score: ', ARI_mbo_1_nor_Lf_Qh_1)
print(' purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', purify_mbo_1_nor_Lf_Qh_1)
print(' inverse purify for MMBO1 normalized L_F & Q_H with \eta =1 : ', inverse_purify_mbo_1_nor_Lf_Qh_1)
print(' NMI for MMBO1 normalized L_F & Q_H with \eta =1 : ', NMI_mbo_1_nor_Lf_Qh_1)



start_time_1_inner_nor_1 = time.time()

# MMBO1 with inner step & sym normalized L_F and gamma=1
u_inner_nor_1,num_repeat_inner_nor = mbo_modularity_inner_step(num_nodes, num_communities, m_1, sym_graph_lap, sym_signless_lap,dt_inner, tol,target_size, inner_step_count)
u_inner_nor_label_1 = vector_to_labels(u_inner_nor_1)
#u_inner_nor_label_dict_1 = label_to_dict(u_inner_nor_label_1)

print("MMBO1 with inner step & normalized L_F and gamma=1:-- %.3f seconds --" % (time.time() - start_time_1_inner_nor_1))
print('MMBO1 with inner step & sym the num_repeat_inner_nor: ',num_repeat_inner_nor)

modularity_1_inner_nor_1 = skn.clustering.modularity(adj_mat,u_inner_nor_label_1,resolution=0.5)
ARI_mbo_1_inner_nor_1 = adjusted_rand_score(u_inner_nor_label_1, sample_labels)
purify_mbo_1_inner_nor_1 = purity_score(sample_labels, u_inner_nor_label_1)
inverse_purify_mbo_1_inner_nor_1 = inverse_purity_score(sample_labels, u_inner_nor_label_1)
NMI_mbo_1_inner_nor_1 = normalized_mutual_info_score(sample_labels, u_inner_nor_label_1)
#f1_mbo_1_inner_nor_1 = f1_score(sample_labels, u_inner_nor_label_1,average='micro')

print(' modularity_1 inner step sym normalized score: ', modularity_1_inner_nor_1)
print(' ARI_1 inner step sym normalized score: ', ARI_mbo_1_inner_nor_1)
print(' purify for MMBO1 inner step with sym normalized \eta =1 : ', purify_mbo_1_inner_nor_1)
print(' inverse purify for MMBO1 inner step with sym normalized \eta =1 : ', inverse_purify_mbo_1_inner_nor_1)
print(' NMI for MMBO1 inner step with sym normalized \eta =1 : ', NMI_mbo_1_inner_nor_1)




start_time_hu_original = time.time()

# test HU original MBO
u_hu_vector, num_iter_HU = mbo_modularity_hu_original(num_nodes, num_communities, m_1,degree, dt_inner, sym_graph_lap, tol,target_size,inner_step_count) 
u_hu_label_1 = vector_to_labels(u_hu_vector)
#u_hu_dict_1 = label_to_dict(u_hu_label_1)

print("HU original MBO:-- %.3f seconds --" % (time.time() - start_time_hu_original))
print('HU original MBO the num_iteration: ', num_iter_HU)

modu_hu_original_1 = skn.clustering.modularity(adj_mat,u_hu_label_1,resolution=0.5)
ARI_hu_original_1 = adjusted_rand_score(u_hu_label_1, sample_labels)
purify_hu_original_1 = purity_score(sample_labels, u_hu_label_1)
inverse_purify_hu_original_1 = inverse_purity_score(sample_labels, u_hu_label_1)
NMI_hu_original_1 = normalized_mutual_info_score(sample_labels, u_hu_label_1)
#f1_hu_original_1 = f1_score(sample_labels, u_hu_label_1, average='macro')

print(' modularity score for HU original MBO: ', modu_hu_original_1)
print(' ARI for HU original MBO: ', ARI_hu_original_1)
print(' purify for HU original MBO : ', purify_hu_original_1)
print(' inverse purify for HU original MBO : ', inverse_purify_hu_original_1)
print(' NMI for HU original MBO : ', NMI_hu_original_1)
#print(' f1 for HU original MBO : ', f1_hu_original_1)



start_time_louvain = time.time()
G = nx.convert_matrix.from_numpy_array(adj_mat)
#G = nx.convert_matrix.from_sparse_matrix(adj_mat)

partition_Louvain = community_louvain.best_partition(G, resolution=0.5)    # returns a dict
louvain_list = list(dict.values(partition_Louvain))    #convert a dict to list
louvain_array = np.asarray(louvain_list)
#print("Louvain:-- %.3f seconds --" % (time.time() - start_time_louvain))

modularity_louvain = skn.clustering.modularity(adj_mat,louvain_array,resolution=1)
ARI_louvain = adjusted_rand_score(louvain_array, sample_labels)
purify_louvain = purity_score(sample_labels, louvain_array)
inverse_purify_louvain = inverse_purity_score(sample_labels, louvain_array)
NMI_louvain = normalized_mutual_info_score(sample_labels, louvain_array)
#f1_louvain = f1_score(y_test, louvain_array,  average='micro')

#print(' modularity Louvain score: ', modularity_louvain)
#print(' ARI Louvain  score: ', ARI_louvain)
#print(' purify for Louvain : ', purify_louvain)
#print(' inverse purify for Louvain : ', inverse_purify_louvain)
#print(' NMI for Louvain with \eta =1 : ', NMI_louvain)
#print(' f1 for Louvain: ', f1_louvain)


start_time_spectral_clustering = time.time()

# Spectral clustering with k-means
sc = SpectralClustering(n_clusters=10, affinity='precomputed')
assignment = sc.fit_predict(adj_mat)

ass_vec = labels_to_vector(assignment)
ass_dict = label_to_dict (assignment)

#print("spectral clustering algorithm:-- %.3f seconds --" % (time.time() - start_time_spectral_clustering))

modularity_spectral_clustering = skn.clustering.modularity(adj_mat,assignment,resolution=0.5)
ARI_spectral_clustering = adjusted_rand_score(assignment, sample_labels)
purify_spectral_clustering = purity_score(sample_labels, assignment)
inverse_purify_spectral_clustering = inverse_purity_score(sample_labels, assignment)
NMI_spectral_clustering = normalized_mutual_info_score(sample_labels, assignment)
#f1_spectral_clustering = f1_score(sample_labels, assignment, average='micro')

#print(' modularity Spectral clustering score(K=10 and m=K): ', modularity_spectral_clustering)
#print(' ARI Spectral clustering  score: ', ARI_spectral_clustering)
#print(' purify for Spectral clustering : ', purify_spectral_clustering)
#print(' inverse purify for Spectral clustering : ', inverse_purify_spectral_clustering)
#print(' NMI for Spectral clustering with \eta =1 : ', NMI_spectral_clustering)
#print(' f1 for Spectral clustering: ', f1_spectral_clustering)
