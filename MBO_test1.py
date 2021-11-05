import numpy as np
import sklearn
import scipy.sparse as spa
from sklearn.metrics.pairwise import pairwise_kernels 
from sklearn.neighbors import kneighbors_graph
import numpy as np
import util
from util import Parameters
from util.build_graph import _graph_params_default_values
from util import misc
from util import BuildGraph
from sklearn.cluster import KMeans
import importlib
importlib.reload(misc)
importlib.reload(util)


class LaplacianClustering(Parameters):
    """ Parameters
        -----------
        N : scalar,
            the number of nodes
        K : scalar,
            the number of clusters
        m : scalar,
            the number of eigenvectors and eigenvalues 
        inner_step_count : int
            for MBO_modularity, Number of times the scheme performs
        dt : float
            stepsize for scheme
        tol : scalar, 
            stopping criterion for iteration
        laplacian_matrix_ : array-like, shape (n_samples, n_samples)
            signed Laplacian matrix
        """
        
    def mbo_modularity_eig(N,K,m,dt,laplacian_matrix_, tol = .5,inner_step_count = 5): # inner stepcount is actually important! and can't be set to 1...
        
        EigVal, EigVec = np.linalg.eig(laplacian_matrix_)
        Val_m = EigVal[m-1]
        Vec_m = EigVec[m-1]
        B_m = np.dot(Vec_m, np.divide(1+dt*Val_m), Vec_m.T)
        U_O = np.zeros((N,K))
        for i in range(1,N):
            k = 1/(K-1)*np.ones((K,1))
            u_init = 
        
        # perform MBO scheme
        n = 0
        U_old = u_init.copy()
        stop_criterion = 0
        while (stop_criterion > tol | n == 200):
            for s in range(inner_step_count):
                U_half_new = np.matmul(B_m, U_old)
                U_old = U_half_new
                s = s + 1 

                U_new = np.zeros((N,K))
                for i in range(1,N):
                    k = np.max(U_half_new[i,:])
                    U_new[i,k] = 1
                
                Ui_diff = []
                Ui_max = []
                for i in range(1,N):    
                    Ui_diff.append((np.linalg.norm(U_new[i,:] - U_old[i,:]))^2)
                    Ui_max.append((np.linalg.norm(U_new[i,:]))^2)

                max_diff = max(Ui_diff)
                max_new = max(Ui_max)
                stop_criterion = max_diff/max_new
            
            n = n + 1

        V_output = []
        for i in range(1,N):
            V_output.append(max(U_new[i,:]))

        return V_output