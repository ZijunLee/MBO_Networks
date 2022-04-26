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


from os import PRIO_PGRP
from joblib import PrintTime
import scipy as sp
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

 
def nystrom_extension(raw_data, num_nystrom=300, gamma=None): # basic implementation
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
    
    m = 10

    if gamma is None:
        print("graph kernel width not specified, using default value 1")
        gamma = 1


    #num_rows=np.shape(adj_mat)[0] 
    #index = permutation(num_rows)
    #other_points = num_rows - num_nystrom
    ##index_sample_remain = np.random.choice(nXx, size=num_nystrom,replace=False)
    #sample_remain = adj_mat[index[:num_nystrom],:]
    ##index_sample = np.random.choice(sample_remain[0], size=num_nystrom,replace=False)
    #print('sample_remain_tranpose shape: ', sample_remain.shape)
    #sample_mat = sample_remain[index_sample,:]
    #print('sample shape: ', sample_mat.shape)
    #A = sample_remain[:,index[:num_nystrom]]     # sampling matrix A
    #B = sample_remain[:,index[num_nystrom:]]     # remaining matrix B
    #samples = np.shape(A)[0]
    #print('sample_mat shape: ', A.shape)
    #print('other_mat shape: ', B.shape)
    #del sample_remain
    
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

    # normalize A and B of W
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    #print('d1: ', d1.shape)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B = B*B1
    #print('B: ', B.shape)


    d1_array = np.expand_dims(d1, axis=-1)
    d2_array = np.expand_dims(d2, axis=-1)
    d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model Q (i.e. Q_A & Q_B)
    #start_time_construct_null_model = time.time()
    total_degree = np.sum(d_array, dtype=np.int64)  
    A_of_null = (d1_array @ d1_array.transpose()) / total_degree
    B_of_null = (d1_array @ d2_array.transpose()) / total_degree
    #print('A_null model: ', A_of_null.shape)
    #print('B_of_null: ', B_of_null.shape)
    #time_null_model = time.time() - start_time_construct_null_model
    #print("construct null model:-- %.3f seconds --" % (time_null_model))

    # normalize A and B of null model Q
    pinv_A_null = pinv(A_of_null)
    B_tranpose_null = B_of_null.transpose()
    d1_null = np.sum(A_of_null,axis = 1) + np.sum(B_of_null,axis = 1)
    #print('d1: ', d1.shape)
    d2_null = np.sum(B_tranpose_null,axis = 1) + np.dot(B_tranpose_null, np.dot(pinv_A_null, np.sum(B_of_null,axis = 1)))
    d_c_null = np.concatenate((d1_null,d2_null),axis = 0)
    dhat_null = np.sqrt(1./d_c_null)
    A_of_null = A_of_null * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = B_of_null * nor_B1


    # do orthogonalization and eigen-decomposition of W
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A + np.dot(np.dot(Asi,BBT),Asi)
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

    # do orthogonalization and eigen-decomposition of null model Q
    B_tranpose_null = B_of_null.transpose()
    Asi_null = sqrtm(pinv(A_of_null))
    BBT_null = np.dot(B_of_null,B_tranpose_null)
    W_null = np.concatenate((A_of_null,B_tranpose_null), axis = 0)
    R_null = A_of_null + np.dot(np.dot(Asi_null,BBT_null),Asi_null)
    R_null = (R_null + R_null.transpose())/2.
    E_null, U_null = eigh(R_null)
    E_null = np.real(E_null)
    ind_null = np.argsort(E_null)[::-1]
    U_null = U_null[:,ind_null]
    E_null = E[ind_null]
    W_null = np.dot(W_null,Asi_null)
    V_null = np.dot(W_null, U_null)
    #V_null = V_null / np.linalg.norm(V_null, axis = 0)
    V_null[index,:] = V_null.copy()
    V_null = np.real(V_null)
    #E_null = 1 + E_null

    E_mix = E + E_null 
    V_mix = V + V_null
    return E_mix, V_mix



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
    
    n_neighbors =10

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
    #distance_matrix = kneighbors_graph(raw_data, n_neighbors=n_neighbors, include_self=True, mode = 'distance')
    #distance_matrix = distance_matrix*distance_matrix # square the distance
    #dist_matrix = 0.5 * (distance_matrix + distance_matrix.T)
    #B = np.exp(-gamma*dist_matrix)

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
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B = B*B1

    # do orthogonalization and eigen-decomposition
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A + np.dot(np.dot(Asi,BBT),Asi)
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



def nystrom_extension_test(raw_data, num_nystrom=300, gamma=None): # basic implementation
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
    
    m = 20
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
    B = rbf_kernel(sample_data, other_data, gamma=gamma)

    # calculating A
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)

    # normalize A and B of W
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    #print('d1: ', d1.shape)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B = B*B1
    #print('B: ', B.shape)


    d1_array = np.expand_dims(d1, axis=-1)
    d2_array = np.expand_dims(d2, axis=-1)
    d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model Q (i.e. Q_A & Q_B)
    #start_time_construct_null_model = time.time()
    total_degree = np.sum(d_array, dtype=np.int64)  
    A_of_null = (d1_array @ d1_array.transpose()) / total_degree
    B_of_null = (d1_array @ d2_array.transpose()) / total_degree
    #print('A_null model: ', A_of_null.shape)
    #print('B_of_null: ', B_of_null.shape)
    #time_null_model = time.time() - start_time_construct_null_model
    #print("construct null model:-- %.3f seconds --" % (time_null_model))

    # normalize A and B of null model Q
    pinv_A_null = pinv(A_of_null)
    B_tranpose_null = B_of_null.transpose()
    d1_null = np.sum(A_of_null,axis = 1) + np.sum(B_of_null,axis = 1)
    #print('d1: ', d1.shape)
    d2_null = np.sum(B_tranpose_null,axis = 1) + np.dot(B_tranpose_null, np.dot(pinv_A_null, np.sum(B_of_null,axis = 1)))
    d_c_null = np.concatenate((d1_null,d2_null),axis = 0)
    dhat_null = np.sqrt(1./d_c_null)
    A_of_null = A_of_null * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = B_of_null * nor_B1

    A = A_of_null + A
    B = B_of_null + B

    # do orthogonalization and eigen-decomposition of W
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A + np.dot(np.dot(Asi,BBT),Asi)
    R = (R+R.transpose())/2.
    E, U = eigh(R)
    E = np.real(E)
    ind = np.argsort(E)[::-1]
    U = U[:,ind]
    E = E[ind]
    W = np.dot(W,Asi)
    V = np.dot(W,np.dot(Asi, U))
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    #E = 1-E
    E = E[:,np.newaxis]

    return E, V