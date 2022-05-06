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
import time
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

def flatten_23(v): # short hand for the swapping axis
    return v.reshape(v.shape[0],-1, order = 'F')

 
def nystrom_extension(raw_data, num_nystrom=300, eta=0.5, gamma=None): # basic implementation
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
    start_time_normalized_W = time.time()
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    #print('d1: ', d1.shape)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    #print('d2 shape: ', d2.shape)
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B = B*B1
    #print('B: ', B.shape)
    print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    #d1_array = np.expand_dims(d1, axis=-1)
    #d2_array = np.expand_dims(d2, axis=-1)
    d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model Q (i.e. Q_A & Q_B)
    #start_time_construct_null_model = time.time()
    total_degree = eta / np.sum(d_array, dtype=np.int64) 
    #A_of_null = (d1_array @ d1_array.transpose()) / total_degree
    #B_of_null = (d1_array @ d2_array.transpose()) / total_degree
    #print('A_null model: ', A_of_null.shape)
    #print('B_of_null: ', B_of_null.shape)
    #print("construct null model:-- %.3f seconds --" % (time.time() - start_time_construct_null_model))

    # normalize A and B of null model Q
    start_time_normalized_Q = time.time()
    #pinv_A_null = pinv(A_of_null)
    #B_tranpose_null = B_of_null.transpose()
    #d1_null = np.sum(A_of_null,axis = 1) + np.sum(B_of_null,axis = 1)
    #print('d1: ', d1.shape)
    #d2_null = np.sum(B_tranpose_null,axis = 1) + np.dot(B_tranpose_null, np.dot(pinv_A_null, np.sum(B_of_null,axis = 1)))
    #d_c_null = np.concatenate((d1_null,d2_null),axis = 0)
    #dhat_null = np.sqrt(1./d_c_null)
    #A_of_null = A_of_null * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    #nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    #B_of_null = B_of_null * nor_B1
    #print("normalized A & B of H:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))
    dhat_null = np.sqrt(d_array)
    A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = total_degree * nor_B1
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))

    # do orthogonalization and eigen-decomposition of W
    start_time_eigendecomposition_W = time.time()
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
    print("orthogonalization and eigen-decomposition of W:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_W))

    # do orthogonalization and eigen-decomposition of null model Q
    start_time_eigendecomposition_Q = time.time()
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
    print("orthogonalization and eigen-decomposition of P:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_Q))

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
    B = rbf_kernel(sample_data, other_data, gamma=gamma)

    # calculating A
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)

    # normalize A and B
    start_time_normalized_W = time.time()
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
    print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))

    # do orthogonalization and eigen-decomposition
    start_time_contruct_S = time.time()
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    R = A + np.dot(np.dot(Asi,BBT),Asi)
    R = (R+R.transpose())/2.
    print("construct matrix S:-- %.3f seconds --" % (time.time() - start_time_contruct_S))
    
    start_time_eigendecomposition_W = time.time()
    E, U = eigh(R)
    print("eigen-decomposition (using SVD):-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_W))
    
    E = np.real(E)
    ind = np.argsort(E)[::-1]
    U = U[:,ind]
    E = E[ind]

    start_time_compute_eigenvectors = time.time()  
    W = np.dot(W,Asi)
    V = np.dot(W, U)
    V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    print("compute eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))

    start_time_compute_eigenvalues = time.time() 
    V = np.real(V)
    E = 1-E
    E = E[:,np.newaxis]
    print("compute eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))
    
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
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=gamma)
    print("calculating W_12:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating A
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)
    print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    start_time_normalized_W = time.time()
    # normalize A and B of W
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
    print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    #d1_array = np.expand_dims(d1, axis=-1)
    #d2_array = np.expand_dims(d2, axis=-1)
    d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model P (i.e. P_A & P_B)
    #start_time_construct_null_model = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    #A_of_null = (d1_array @ d1_array.transpose()) / total_degree
    #B_of_null = (d1_array @ d2_array.transpose()) / total_degree
    #print('A_null model: ', A_of_null.shape)
    #print('B_of_null: ', B_of_null.shape)
    #time_null_model = time.time() - start_time_construct_null_model
    #print("construct null model:-- %.3f seconds --" % (time_null_model))

    start_time_normalized_Q = time.time()
    # normalize A and B of null model Q
    #pinv_A_null = pinv(A_of_null)
    #B_tranpose_null = B_of_null.transpose()
    #d1_null = np.sum(A_of_null,axis = 1) + np.sum(B_of_null,axis = 1)
    #print('d1: ', d1.shape)
    #d2_null = np.sum(B_tranpose_null,axis = 1) + np.dot(B_tranpose_null, np.dot(pinv_A_null, np.sum(B_of_null,axis = 1)))
    #d_c_null = np.concatenate((d1_null,d2_null),axis = 0)
    #dhat_null = np.sqrt(1./d_c_null)
    #A_of_null = A_of_null * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    #nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    #B_of_null = B_of_null * nor_B1

    dhat_null = np.sqrt(d_c)
    A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = total_degree * nor_B1
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))

    start_time_construct_new_R11_R12 = time.time()
    A =  A - A_of_null
    B =  B - B_of_null
    print("construct R_11 & R_12:-- %.3f seconds --" % (time.time() - start_time_construct_new_R11_R12))

    # do orthogonalization and eigen-decomposition of W
    start_time_contruct_S = time.time()     
    B_T = B.transpose()
    Asi = sqrtm(pinv(A))
    BBT = np.dot(B,B_T)
    W = np.concatenate((A,B_T), axis = 0)
    S = A + np.dot(np.dot(Asi,BBT),Asi)
    S = (S+S.transpose())/2.
    print("construct matrix S:-- %.3f seconds --" % (time.time() - start_time_contruct_S))
    
    start_time_eigendecomposition_W = time.time()
    #E, U = eigh(S)
    U, E, vt = sp.linalg.svd(S, full_matrices=False)
    #U = U[:,:k]
    #E = E[:k]
    #vt = vt[:k, :]
    print("eigen-decomposition (using SVD):-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_W))

    E = np.real(E)
    ind = np.argsort(E)[::-1]
    U = U[:,ind]
    E = E[ind]

    start_time_compute_eigenvectors = time.time()    
    W = np.dot(W,Asi)
    M = np.dot(U, np.diag(1. / np.sqrt(E)))
    #V = np.dot(W,np.dot(Asi, U))
    V = np.dot(W, M)
    #V = V / np.linalg.norm(V, axis = 0)
    V[index,:] = V.copy()
    V = np.real(V)
    print("compute eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    
    start_time_compute_eigenvalues = time.time() 
    E = 2-E
    E = E[:,np.newaxis]
    print("compute eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))

    return E, V



def nystrom_QR(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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


    # calculating W_21
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=gamma)
    print("calculating W_21:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating W_11
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)
    print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    # construct null model P
    start_time_construct_P = time.time()
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_array = np.concatenate((d1,d2), axis=None)
    
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    dhat_null = np.sqrt(d_array)
    A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = total_degree * nor_B1
    print("construct null model P:-- %.3f seconds --" % (time.time() - start_time_construct_P))
    
    # compute B = W -P
    start_time_construct_B = time.time()
    A = A - A_of_null
    B_T = B_T - B_of_null.transpose()
    print("compute B:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    start_time_approximation_B = time.time()
    pinv_A_new = pinv(A)
    d2_new = np.dot(B_T, np.dot(pinv_A_new, np.sum(B_T.transpose(),axis = 1)))
    d_inverse = np.sqrt(1./d2_new)
    d_inverse = np.expand_dims(d_inverse, axis=-1)
    B_T = B_T * d_inverse
    print("computing the approximation of B_21:-- %.3f seconds --" % (time.time() - start_time_approximation_B))

    # QR decomposition for the approximation of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(B_T,mode='reduced')
    print("QR decomposition for the approximation of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_A_new, R.transpose()))
    S = (S+S.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    
    #d_c = np.concatenate((d1,d2),axis = 0)
    #dhat = np.sqrt(1./d_c)
    #A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_nystrom+other_points,np.newaxis].transpose())
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    #B = B*B1
    #print("normalized A & B of F:-- %.3f seconds --" % (time.time() - start_time_normalized_W))

    # do orthogonalization and eigen-decomposition
    #start_time_eigendecomposition_W = time.time()
    #B_T = B.transpose()
    #Asi = sqrtm(pinv(A))
    #BBT = np.dot(B,B_T)
    #W = np.concatenate((A,B_T), axis = 0)
    #R = A + np.dot(np.dot(Asi,BBT),Asi)
    #R = (R+R.transpose())/2.
    E, U = eigh(S)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

    E = np.real(E)
    ind = np.argsort(E)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U = U[:,ind]
    E = E[ind]
    V = np.dot(Q, U)
    V = V / np.linalg.norm(V, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V = np.real(V)

    # calculating eigenvalues
    start_time_compute_eigenvalues = time.time()
    E = 2-E
    E = E[:,np.newaxis]
    print("calculating eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))
    
    return E,V, other_data, index


def nystrom_QR_l_sym(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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


    # calculating W_21
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=gamma)
    print("calculating W_21:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating W_11
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)
    print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    # computing the approximation of W_21
    start_time_approximation_W21 = time.time()
    pinv_A = pinv(A)
    B_T = B.transpose()
    d2 = np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_inverse = np.sqrt(1./d2)
    d_inverse = np.expand_dims(d_inverse, axis=-1)
    B_T = B_T * d_inverse
    print("computing the approximation of W_21:-- %.3f seconds --" % (time.time() - start_time_approximation_W21))
    
    # QR decomposition for the approximation of W_21
    start_time_QR_decomposition_approximation_W21 = time.time()
    Q, R = np.linalg.qr(B_T,mode='reduced')
    print("QR decomposition for the approximation of W_21:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_W21))
    
    # construct S
    start_time_construct_S = time.time()    
    S = np.dot(R, np.dot(pinv_A, R.transpose()))
    S = (S+S.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E, U = eigh(S)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))
    
    E = np.real(E)
    ind = np.argsort(E)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U = U[:,ind]
    E = E[ind]
    V = np.dot(Q, U)
    V = V / np.linalg.norm(V, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V = np.real(V)

    # calculating eigenvalues
    start_time_compute_eigenvalues = time.time()
    E = 1-E
    E = E[:,np.newaxis]
    print("calculating eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))

    return E,V, other_data, index