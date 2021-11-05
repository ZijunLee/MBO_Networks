import numpy as np
import numpy as np
import util
from util import Parameters
from util import misc
import importlib
importlib.reload(misc)
importlib.reload(util)


class LaplacianClustering(Parameters):
    """ Parameters
        -----------
        N : scalar,
            the number of nodes
        K : scalar,
            the number of clusters (note: N >> K)
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
        U_0 = np.zeros((N,K))
        for i in range(1,N):
            k = np.random.random_integers(K)
            U_0[i-1,k-1] = 1
            U_init = U_0
        
        # perform MBO scheme
        n = 0
        U_old = U_init.copy()
        stop_criterion = 0
        while (stop_criterion > tol | n == 200):
            for s in range(inner_step_count):
                U_half_new = np.matmul(B_m, U_old)
                U_old = U_half_new
                s = s + 1 

                U_new = np.zeros((N,K))
                for i in range(1,N):
                    k = np.max(U_half_new[i-1,:])
                    U_new[i-1,k-1] = 1
                
                Ui_diff = []
                Ui_max = []
                for i in range(1,N):    
                    Ui_diff.append((np.linalg.norm(U_new[i-1,:] - U_old[i-1,:]))^2)
                    Ui_max.append((np.linalg.norm(U_new[i-1,:]))^2)

                max_diff = max(Ui_diff)
                max_new = max(Ui_max)
                stop_criterion = max_diff/max_new
            
            n = n + 1

        V_output = []
        for i in range(1,N):
            V_output.append(max(U_new[i-1,:]))

        return V_output