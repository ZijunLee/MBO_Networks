import os
from sklearn.decomposition import PCA
import time
from graph_cut.data.read_mnist import Read_mnist_function, subsample
from graph_cut_util import build_affinity_matrix_new
from MBO_Network import adj_to_laplacian_signless_laplacian,construct_null_model


## parameter setting
num_communities = 10
m = 1 * num_communities

gpath = '/'.join(os.getcwd().split('/')[:-1])

#full_data, full_labels = Read_mnist_function(digits = range(10),path = gpath+'/MBO_signed_graphs/graph_cut/data')
full_data, full_labels = Read_mnist_function(digits = range(10),path ='/home/zijul93/MBO_SignedNetworks/graph_cut/data')
full_data = full_data/255.

sample_data,sample_labels = subsample(sample_num = 1500, rd = full_data, labels = full_labels)

pca = PCA(n_components = 50, svd_solver='full')
pca.fit_transform(full_data)
train_data = pca.transform(sample_data)

del full_data

n1, p = train_data.shape

gamma = 1. / p

adj_mat = build_affinity_matrix_new(train_data,gamma=gamma, affinity='rbf',n_neighbors=10, neighbor_type='knearest')
#print('adj_mat shape: ', adj_mat.shape)

del train_data

null_model = construct_null_model(adj_mat)

num_nodes, m_1, degree, target_size, graph_laplacian, sym_graph_lap,rw_graph_lap, signless_laplacian, sym_signless_lap, rw_signless_lap = adj_to_laplacian_signless_laplacian(adj_mat, null_model, num_communities,m ,target_size=None)

del null_model

print('symmetric normalized L_F shape: ', sym_graph_lap.shape)
print('symmetric normalized Q_H shape: ', sym_signless_lap.shape)

# Compute L_{mix} = L_{F_sym} + Q_{H_sym}
start_time_l_mix = time.time()
l_mix = sym_graph_lap + sym_signless_lap
print("compute l_{mix}:-- %.3f seconds --" % (time.time() - start_time_l_mix))