#--------------------------------------------------------------------------
# Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
#
# This file is part of the diffuse-interface graph algorithm code. 
# There are currently no licenses. 
#
#--------------------------------------------------------------------------
# Description: 
#
#           Implementation of Nystrom Extension.          
#
#--------------------------------------------------------------------------



import scipy.sparse as spa
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from scipy.spatial.distance import cdist
from scipy.linalg import pinv
from scipy.linalg import sqrtm  
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.linalg import svd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

def flatten_23(v): # short hand for the swapping axis
    return v.reshape(v.shape[0],-1, order = 'F')

 
def nystrom_extension(adj_mat, num_nystrom=300, gamma=None): # basic implementation
    """ Nystrom Extension


    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.

    sigma : width of the rbf kernel

    num_nystrom : int, 
            number of sample points 

    Return 
    ----------
    V : eigenvectors
    E : eigenvalues    
    """

    #format data to right dimensions
    # width = int((np.sqrt(raw_data.shape[1]/n_channels)-1)/2)
    # if kernel_flag: # spatial kernel involved   # depreciated. Put spatial mask in the image patch extraction process
    #     kernel = make_kernel(width = width, n_channels = n_channels)
    #     scale_sqrt = np.sqrt(kernel).reshape(1,len(kernel))
    if gamma is None:
        print("graph kernel width not specified, using default value 1")
        gamma = 1

    #n_neighbors =10
    nXx=np.shape(adj_mat)[0] 
    index = permutation(nXx)
    other_points = nXx - num_nystrom
    #index_sample_remain = np.random.choice(nXx, size=num_nystrom,replace=False)
    sample_remain = adj_mat[index[:num_nystrom],:]
    #index_sample = np.random.choice(sample_remain[0], size=num_nystrom,replace=False)
    #print('sample_remain_tranpose shape: ', sample_remain.shape)
    #sample_mat = sample_remain[index_sample,:]
    #print('sample shape: ', sample_mat.shape)
    A = sample_remain[:,index[:num_nystrom]]     # sampling matrix A
    B = sample_remain[:,index[num_nystrom:]]     # remaining matrix B
    samples = np.shape(A)[0]
    print('sample_mat shape: ', A.shape)
    print('other_mat shape: ', B.shape)
    del sample_remain
    
    #sumw=np.sum(adj_mat,axis=1)  
    #sumw=np.vstack((sumw,[0 for i in range(nXx)])).T  
    #idsumw=np.lexsort([-sumw[:,0]])  
    #X1,X2,AA,BB=[],[],[],[]  
    #for ii in range(nXx):
    #    i=idsumw[ii]
    #    if sumw[i,1]==0: 
    #        X1.extend([i])  
    #        sim1=[0 for k1 in range(len(X1))]
    #        sim2=[0 for k2 in range(len(X2)+n_neighbors)] 
    #        for j in range(n_neighbors):
    #            if adj_mat[i,j] in X1:
    #                id1=X1.index(adj_mat[i,j]) 
    #                sim1[id1]=adj_mat[i,j]
    #                sim2.pop()  
    #            elif adj_mat[i,j] in X2:
    #                id2=X2.index(adj_mat[i,j])
    #                sim2[id2]=adj_mat[i,j]
    #                sim2.pop()
    #            else:
    #                X2.extend([adj_mat[i,j]])  
    #                sim2[len(X2)-1]=adj_mat[i,j]
    #        #        sumw[adj_mat[i,j],1]=-1
    #        BB.append(sim2)
            
    #del distances,indices
    #gc.collect() 
    #samples=len(X1)
    #remains=len(X2)
    #A=np.eye(samples)  
    #B=np.zeros((samples,remains))  
    #for i in range(samples):
    #    A[i,:(i+1)]=AA[i]
    #    B[i,:len(BB[i])]=BB[i]
    #del AA,BB

    #seed=44
    #rng = np.random.RandomState(seed)
    #num_rows = affinity_matrix_.shape[0] 
    #index = rng.choice(num_rows, num_nystrom)  
    #index = permutation(num_rows)
    #if num_nystrom == None:
    #    raise ValueError("Please Provide the number of sample points in num_nystrom")
    #sample_data = raw_data[index[:num_nystrom]]
    #other_data = raw_data[index[num_nystrom:]]
    #sample_mat = affinity_matrix_[index, :]


    # calculating B
    #other_points = num_rows - num_nystrom
    #distb = cdist(sample_data,other_data,'sqeuclidean')
    #if gamma == None:
    #    gamma = np.percentile(np.percentile(distb, axis = 1, q = 5),q = 40) # a crude automatic kernel
    #B = np.exp(-distb/gamma).astype(np.float32)    

    # calculating A
    #dista = cdist(sample_data,sample_data,'sqeuclidean')
    #dista = rbf_kernel(sample_data, sample_data, gamma=gamma)
    #A = np.exp(-dista/gamma).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    #dd = np.dot(dhat.reshape((len(dhat),1)),dhat.reshape((1,len(dhat))))
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points, np.newaxis].transpose())
    B = B*B1
    #A=A*dd[:samples,:samples]
    #B=B*dd[:samples,samples:]

    # do orthogonalization and eigen-decomposition
    B_T = B.transpose()
    #A_mat_nan = np.isnan(A).any()
    #A_mat_inf = np.isinf(A).any()
    #print('A_mat_nan: ',A_mat_nan)
    #print('A_mat_inf: ', A_mat_inf)
    A = np.nan_to_num(A)
    Asi = sqrtm(pinv(A))
    #detA=np.linalg.det(np.sqrt(A))   
    #if detA>0:
    #    Asi=np.linalg.inv(np.sqrt(A))   
    #else:
    #    Asi=np.linalg.pinv(np.sqrt(A))  
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A+ np.dot(np.dot(Asi,BBT),Asi)
    R = (R+R.transpose())/2.
    #R_mat_nan = np.isnan(R).any()
    #R_mat_inf = np.isinf(R).any()
    #print('X_mat_nan: ',R_mat_nan)
    #print('X_mat_inf: ', R_mat_inf)
    R = np.nan_to_num(R)
    E, U = eigh(R)
    E = np.real(E)
    ind = np.argsort(E)
    U = U[:,ind]
    #E = E[ind]
    W = np.dot(W,Asi)
    V = np.dot(W, U)
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    #E = 1-E
    #E = E[:,np.newaxis]
    
    #E = np.real(E)
    #E =sorted(E)
    #ind = np.argsort(E)[::-1]
    #U = U[:,ind]
    #E = E[ind]
    #W = np.dot(W,Asi)
    #V = np.dot(W, U)
    #V = V / np.linalg.norm(V, axis = 0)
    #V = normalize(V)
    #V[index,:] = V.copy()
    #V = np.real(V)
    #E = E-1
    #E = E[:,np.newaxis]
    return E,V



def nystrom_new(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
    """ Nystrom Extension


    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.

    sigma : width of the rbf kernel

    num_nystrom : int, 
            number of sample points 

    Return 
    ----------
    V : eigenvectors
    E : eigenvalues    
    """

    if gamma is None:
        print("graph kernel width not specified, using default value 1")
        gamma = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]


    # calculating B
    other_points = num_rows - num_nystrom
    B = rbf_kernel(sample_data, other_data, gamma=gamma)

    # calculating A
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B = B*B1

    # do orthogonalization and eigen-decomposition
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A+ np.dot(np.dot(Asi,BBT),Asi)
    R = (R+R.transpose())/2.
    E, U = eigh(R)
    E = np.real(E)
    ind = np.argsort(E)[::-1]
    U = U[:,ind]
    E = E[ind]
    W = np.dot(W,Asi)
    V = np.dot(W, U)
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    E = 1-E
    E = E[:,np.newaxis]
    return E,V
