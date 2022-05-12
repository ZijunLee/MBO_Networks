import os
import sys
import graspologic
from graph_cut.util.nystrom import nystrom_QR_l_sym
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from turtle import color
import matplotlib.pyplot as plt
from matplotlib import lines
import math
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.decomposition import PCA
import graphlearning as gl
from graph_mbo.utils import get_fidelity_term,vector_to_labels,label_to_dict
import time
from numpy.random import permutation
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import pinv, eigh, sqrtm
from graph_cut.data.read_mnist import Read_mnist_function, subsample
from graspologic.plot import heatmap
from graspologic.simulations import sbm


data, gt_labels = gl.datasets.load('mnist')
pca = PCA(n_components = 50,svd_solver='full')
Z_training = pca.fit_transform(data)
W = gl.weightmatrix.knn(Z_training, 10)
G = nx.convert_matrix.from_scipy_sparse_matrix(W)

plt.figure(figsize=(8,8))
subax1 = plt.subplot(121)
nx.draw_networkx(G, with_labels=False, font_weight='bold')
#subax2 = plt.subplot(122)
#nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.savefig("path.png")
plt.show() 