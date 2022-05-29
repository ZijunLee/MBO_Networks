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


from regex import P
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

    dhat_null = np.sqrt(d_array)
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
    #V = np.dot(W, U)
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

    # normalized W 
    start_time_normalized_W = time.time()
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_array = np.concatenate((d1,d2), axis=None)
    #d_c = np.concatenate((d1,d2),axis = 0)
    #dhat = np.sqrt(1./d_array)
    d_inverse = np.sqrt(1./d2)
    d_inverse = np.expand_dims(d_inverse, axis=-1)
    #A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    #print('W_11: ', A.shape)
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_T = B_T * d_inverse
    #B_T = B.transpose() 
    #print('W_21: ', B_T.shape) 
    print("normalized W:-- %.3f seconds --" % (time.time() - start_time_normalized_W))  

    # construct null model P & normalized P
    start_time_construct_P = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    A_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[0:num_nystrom,np.newaxis].transpose()) 
    B_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null_T = B_of_null.transpose() * d_inverse
    #dhat_null = np.sqrt(d_array)
    #A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    #print('P_11: ', A_of_null.shape)
    #nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    #B_of_null = total_degree * nor_B1
    #print('P_12: ', B_of_null.shape)
    #A_of_null = total_degree * (np.dot(d_array[0:num_nystrom,np.newaxis],d_array[0:num_nystrom,np.newaxis].transpose()))
    #print('P_11: ', A_of_null.shape)
    #nor_B1 = np.dot(d_array[0:num_nystrom,np.newaxis], d_array[num_nystrom:num_rows,np.newaxis].transpose())
    #B_of_null = total_degree * nor_B1
    print("construct null model P & normalized P:-- %.3f seconds --" % (time.time() - start_time_construct_P))
    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    A = A - A_of_null
    B_T = B_T - B_of_null_T
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    pinv_A_new = pinv(A)
    #d2_new = np.dot(B_T, np.dot(pinv_A_new, np.sum(B_T.transpose(),axis = 1)))
    #d_inverse = np.sqrt(1./d2_new)
    #d_inverse = np.expand_dims(d_inverse, axis=-1)
    #print('B_T: ', B_T.shape)
    #print('d_inverse: ', d_inverse.shape)
    #B_T = B_T * d_inverse
    #print("computing the approximation of B_21:-- %.3f seconds --" % (time.time() - start_time_approximation_B))

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(B_T,mode='reduced')
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_A_new, R.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
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
    
    #rw_left_eigvec = V * d_inverse
    #rw_right_eugvec = V * np.sqrt(d2_new)
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
    print('start Nystrom QR decomposition for L_sym / L_rw')    

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
    d2_expand = np.expand_dims(d2, axis=-1)
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
    E = 1-E
    E = E[:,np.newaxis]
    V = np.dot(Q, U)
    V = V / np.linalg.norm(V, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V = np.real(V)

    start_time_compute_eigenvector_rw = time.time()
    rw_left_eigvec = V * d_inverse
    rw_right_eugvec = V * np.sqrt(d2_expand)
    print("calculating rw eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvector_rw))

    return E,V, other_data, index, rw_left_eigvec, rw_right_eugvec


def nystrom_QR_B_signed(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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

    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_array = np.concatenate((d1,d2), axis=None)

    # construct null model P 
    start_time_construct_P = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    A_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[0:num_nystrom,np.newaxis].transpose()) 
    B_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[num_nystrom:num_rows,np.newaxis].transpose())
    print("construct null model P:-- %.3f seconds --" % (time.time() - start_time_construct_P))

    # compute B = W - P
    start_time_construct_B = time.time()
    B_11 = A - A_of_null
    B_12 = B - B_of_null
    print("compute B:-- %.3f seconds --" % (time.time() - start_time_construct_B))

    # positive part of B, i.e. B_{11}^+ & B_{12}^+
    B_11_positive = np.where(B_11 > 0, B_11, 0)   # B_{11}^+
    B_12_positive = np.where(B_12 > 0, B_12, 0)   # B_{12}^+

    # negative part of B, i.e. B_{11}^- & B_{12}^-
    B_11_negative = -np.where(B_11 < 0, B_11, 0)   # B_{11}^-
    B_12_negative = -np.where(B_12 < 0, B_12, 0)   # B_{12}^-

    # normalized the positive part of B, i.e. B^+
    start_time_normalized_B_positive = time.time()
    pinv_B11_positive = pinv(B_11_positive)
    B12_positive_T = B_12_positive.transpose()
    #d1_posi = np.sum(B_11_positive,axis = 1) + np.sum(B_12_positive,axis = 1)
    d2_posi = np.dot(B12_positive_T, np.dot(pinv_B11_positive, np.sum(B_12_positive,axis = 1)))
    #d_positive_array = np.concatenate((d1_posi,d2_posi), axis=None)
    d_posi_inverse = np.sqrt(1./d2_posi)
    d_posi_inverse = np.nan_to_num(d_posi_inverse)
    d_posi_inverse = np.expand_dims(d_posi_inverse, axis=-1)
    B12_positive_T = B12_positive_T * d_posi_inverse
    #B_T = B.transpose() 
    #print('W_21: ', B_T.shape) 
    print("normalized B^+:-- %.3f seconds --" % (time.time() - start_time_normalized_B_positive))  

    # normalized the negative part of B, i.e. B^-
    start_time_normalized_B_negative = time.time()
    pinv_B11_neg = pinv(B_11_negative)
    B12_neg_T = B_12_negative.transpose()
    #d1_neg = np.sum(B_11_negative,axis = 1) + np.sum(B_12_negative,axis = 1)
    d2_neg = np.dot(B12_neg_T, np.dot(pinv_B11_neg, np.sum(B_12_negative,axis = 1)))
    #d_neg_array = np.concatenate((d1_neg,d2_neg), axis=None)
    d_neg_inverse = np.sqrt(1./d2_neg)
    d_neg_inverse = np.nan_to_num(d_neg_inverse)
    d_neg_inverse = np.expand_dims(d_neg_inverse, axis=-1)
    B12_neg_T = B12_neg_T * d_neg_inverse
    print("normalized B^-:-- %.3f seconds --" % (time.time() - start_time_normalized_B_negative))
    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_11 = B_11
    M_21 = B12_positive_T - B12_neg_T
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    pinv_M_11 = pinv(M_11)
    #d2_new = np.dot(B_T, np.dot(pinv_A_new, np.sum(B_T.transpose(),axis = 1)))
    #d_inverse = np.sqrt(1./d2_new)
    #d_inverse = np.expand_dims(d_inverse, axis=-1)
    #print('B_T: ', B_T.shape)
    #print('d_inverse: ', d_inverse.shape)
    #B_T = B_T * d_inverse
    #print("computing the approximation of B_21:-- %.3f seconds --" % (time.time() - start_time_approximation_B))

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(M_21,mode='reduced')
    print("QR decomposition of M_{FH}:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_M_11, R.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
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
    
    #rw_left_eigvec = V * d_inverse
    #rw_right_eugvec = V * np.sqrt(d2_new)
    return E,V, other_data, index


def nystrom_QR_1(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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

    start_time_normalized_W = time.time()
    # normalize A and B of W
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    A = A*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
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

    dhat_null = np.sqrt(d_array)
    A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = total_degree * nor_B1
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))

    #start_time_construct_new_R11_R12 = time.time()
    #A =  A - A_of_null
    #B =  B - B_of_null
    #print("construct R_11 & R_12:-- %.3f seconds --" % (time.time() - start_time_construct_new_R11_R12))

    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    A =  A - A_of_null
    B =  B - B_of_null
    #print('B: ', B.shape)
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    pinv_A_new = pinv(A)
    B_T = B.transpose()
    #d2_new = np.dot(B_T, np.dot(pinv_A_new, np.sum(B_T.transpose(),axis = 1)))
    #d2_new = np.expand_dims(d2_new, axis=-1)
    #d_inverse = np.sqrt(1./d2_new)
    #d_inverse = np.nan_to_num(d_inverse)
    #print('B_T: ', B_T.shape)
    #print('d_inverse: ', d_inverse.shape)
    #B_T = B_T * d_inverse
    #M_11 = B[:,:num_nystrom]
    #M_12 = B[:,num_nystrom:]
    #print('M_11: ', M_11.shape)
    #print('M_21: ',M_12.shape)
    #pinv_A_new = pinv(M_11)
    #print("computing the approximation of B_21:-- %.3f seconds --" % (time.time() - start_time_approximation_B))


    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(B_T, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_A_new, R.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
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
    #print('E: ', E.shape)
    print("calculating eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))
    
    #rw_left_eigvec = V * d_inverse
    #rw_right_eugvec = V * np.sqrt(d2_new)
    return E,V, other_data, index



def nystrom_QR_1_signed(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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

    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_array = np.concatenate((d1,d2), axis = 0)
    #dhat_B = np.sqrt(1./d_array)

    # construct null model P 
    start_time_construct_P = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    A_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[0:num_nystrom,np.newaxis].transpose()) 
    B_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[num_nystrom:num_rows,np.newaxis].transpose())
    print("construct null model P:-- %.3f seconds --" % (time.time() - start_time_construct_P))

    # compute B = W - P
    start_time_construct_B = time.time()
    B_11 = A - A_of_null
    B_12 = B - B_of_null
    print("compute B:-- %.3f seconds --" % (time.time() - start_time_construct_B))

    # positive part of B, i.e. B_{11}^+ & B_{12}^+
    B_11_positive = np.where(B_11 > 0, B_11, 0)   # B_{11}^+
    B_12_positive = np.where(B_12 > 0, B_12, 0)   # B_{12}^+

    # negative part of B, i.e. B_{11}^- & B_{12}^-
    B_11_negative = -np.where(B_11 < 0, B_11, 0)   # B_{11}^-
    B_12_negative = -np.where(B_12 < 0, B_12, 0)   # B_{12}^-


    # normalize B^+
    start_time_normalized_W = time.time()
    pinv_A_pos = pinv(B_11_positive)
    B_T_pos = B_12_positive.transpose()
    d1 = np.sum(B_11_positive,axis = 1) + np.sum(B_12_positive,axis = 1)
    d2 = np.sum(B_T_pos,axis = 1) + np.dot(B_T_pos, np.dot(pinv_A_pos, np.sum(B_12_positive,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    dhat = np.nan_to_num(dhat)
    print('degree of B^+: ', dhat)
    B_11_positive = B_11_positive*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_12_positive = B_12_positive * B1 
    #B_12_positive = np.nan_to_num(B_12_positive)   
    print("normalized B^+:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    # normalize B^-
    start_time_normalized_W = time.time()
    pinv_A_neg = pinv(B_11_negative)
    B_T_neg = B_12_negative.transpose()
    d1 = np.sum(B_11_negative,axis = 1) + np.sum(B_12_negative,axis = 1)
    d2 = np.sum(B_T_neg,axis = 1) + np.dot(B_T_neg, np.dot(pinv_A_neg, np.sum(B_12_negative,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = np.sqrt(1./d_c)
    dhat = np.nan_to_num(dhat)
    print('degree of B^-: ', dhat)
    B_11_negative = B_11_negative*(np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_12_negative = B_12_negative * B1
    #B_12_negative = np.nan_to_num(B_12_negative)    
    print("normalized B^-:-- %.3f seconds --" % (time.time() - start_time_normalized_W))

    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    A = B_11_positive - B_11_negative
    B =  B_12_positive - B_12_negative
    #print('B: ', B.shape)
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    pinv_A_new = pinv(A)
    B_T = B.transpose()

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(B_T, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_A_new, R.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
    S = (S+S.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    #S = np.nan_to_num(S)
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
    E = 2 - E
    E = E[:,np.newaxis]
    E = E[np.nonzero(E > 0)]
    #print('E: ', E.shape)
    print("calculating eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))

    return E,V, other_data, index




def nystrom_QR_1_random_walk(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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

    start_time_normalized_W = time.time()
    # normalize W_11 & W_12
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    dhat = 1./d_c
    dhat = np.expand_dims(dhat, axis=-1)
    #dhat_list = dhat.tolist()
    #print('dhat_list: ', dhat_list)
    #D_A = np.diag(dhat_list[0:num_nystrom])
    #print('D_A: ', D_A.shape)
    A = A * dhat[:num_nystrom]
    #D_B = np.diag(dhat[num_nystrom:num_rows])
    #B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B = B * dhat[num_nystrom:num_rows].transpose()  
    print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    d1_array = np.expand_dims(d1, axis=-1)
    d2_array = np.expand_dims(d2, axis=-1)
    d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model P (i.e. P_A & P_B)
    #start_time_construct_null_model = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    A_of_null = (d1_array @ d1_array.transpose()) * total_degree
    B_of_null = (d1_array @ d2_array.transpose()) * total_degree
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
    #dhat_null = 1./d_c_null
    A_of_null = A_of_null * dhat[0:num_nystrom]
    #nor_B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null = B_of_null * dhat[num_nystrom:num_rows].transpose() 

    #dhat_null = np.sqrt(d_array)
    #A_of_null = total_degree * (np.dot(dhat_null[0:num_nystrom,np.newaxis],dhat_null[0:num_nystrom,np.newaxis].transpose()))
    #nor_B1 = np.dot(dhat_null[0:num_nystrom,np.newaxis], dhat_null[num_nystrom:num_rows,np.newaxis].transpose())
    #B_of_null = total_degree * nor_B1
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))
    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    A =  A - A_of_null
    B =  B - B_of_null
    #print('B: ', B.shape)
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    pinv_A_new = pinv(A)
    B_T = B.transpose()
    #d2_new = np.dot(B_T, np.dot(pinv_A_new, np.sum(B_T.transpose(),axis = 1)))
    #d2_new = np.expand_dims(d2_new, axis=-1)
    #d_inverse = np.sqrt(1./d2_new)
    #d_inverse = np.nan_to_num(d_inverse)
    #print('B_T: ', B_T.shape)
    #print('d_inverse: ', d_inverse.shape)
    #B_T = B_T * d_inverse
    #M_11 = B[:,:num_nystrom]
    #M_12 = B[:,num_nystrom:]
    #print('M_11: ', M_11.shape)
    #print('M_21: ',M_12.shape)
    #pinv_A_new = pinv(M_11)
    #print("computing the approximation of B_21:-- %.3f seconds --" % (time.time() - start_time_approximation_B))


    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q, R = np.linalg.qr(B_T, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S = np.dot(R, np.dot(pinv_A_new, R.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
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
    #print('E: ', E.shape)
    print("calculating eigenvalues:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvalues))

    return E,V, other_data, index



def nystrom_QR_1_sym_rw(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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
    print('start Nystrom QR decomposition for L_mix_sym / L_mix_rw')     

    if gamma is None:
        print("graph kernel width not specified, using default value 1")
        gamma = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    other_rows = num_rows - num_nystrom


    # calculating W_21
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=gamma)
    print("calculating W_21:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating W_11
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=gamma)
    print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    #d_c = np.concatenate((d1,d2),axis = 0)
    d_c = np.concatenate((d1,d2), axis=None)
    
    total_degree = 1. / np.sum(d_c, dtype=np.int64)
    A_of_null = total_degree * (d_c[0:num_nystrom,np.newaxis] @ d_c[0:num_nystrom,np.newaxis].transpose()) 
    B_of_null = total_degree * (d_c[0:num_nystrom,np.newaxis] @ d_c[num_nystrom:num_rows,np.newaxis].transpose())
    
    start_time_random_walk = time.time()
    # random walk W_11 & W_12
    start_time_normalized_W = time.time()
    dhat = 1./d_c
    id_mat = np.ones(num_nystrom)
    id_mat = np.expand_dims(id_mat, axis=-1)
    #print('id_mat: ', id_mat.shape)
    #id_mat_other = np.ones(other_rows)
    #id_mat_other = np.expand_dims(id_mat_other, axis=-1)
    W_11_rw =  A * dhat[0:num_nystrom,np.newaxis].transpose() 
    W_12_rw =  B * dhat[num_nystrom:num_rows,np.newaxis].transpose()
    print("random walk W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    #d1_array = np.expand_dims(d1, axis=-1)
    #d2_array = np.expand_dims(d2, axis=-1)
    #d_array = np.concatenate((d1,d2), axis=None)
    #print('d_array: ', type(d_array))
    #dergee_di_null = np.expand_dims(d_array, axis=-1)
    
    # construct A & B of null model P (i.e. P_A & P_B)
    #start_time_construct_null_model = time.time()

    #A_of_null = (np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose())) * total_degree
    #B_of_null = (dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose()) * total_degree

    start_time_normalized_Q = time.time()
    A_of_null_rw = A_of_null * dhat[0:num_nystrom,np.newaxis].transpose()
    B_of_null_rw = B_of_null * dhat[num_nystrom:num_rows,np.newaxis].transpose()
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))
    
    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_11_rw =  W_11_rw - A_of_null_rw
    M_12_rw =  W_12_rw - B_of_null_rw
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    pinv_A_new_rw = pinv(M_11_rw)
    M_21_rw = M_12_rw.transpose()

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_rw, R_rw = np.linalg.qr(M_21_rw, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_rw = np.dot(R_rw, np.dot(pinv_A_new_rw, R_rw.transpose()))
    S_rw = (S_rw + S_rw.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_rw, U_rw = eigh(S_rw)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

    E_rw = np.real(E_rw)
    ind_rw = np.argsort(E_rw)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U_rw = U_rw[:,ind_rw]
    E_rw = E_rw[ind_rw]
    E_rw = 2 - E_rw
    E_rw = E_rw[:,np.newaxis]
    V_rw = np.dot(Q_rw, U_rw)
    V_rw = V_rw / np.linalg.norm(V_rw, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_rw = np.real(V_rw)

    print("random walk L_{mix}:-- %.3f seconds --" % (time.time() - start_time_random_walk))

    start_time_symmetric = time.time()
    # symmetric normalize A and B of W 
    start_time_normalized_W = time.time()
    dhat = np.sqrt(1./d_c)
    W_11_sym = A * (np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    W_12_sym = B * B1    
    print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))

    #d_array = np.concatenate((d1,d2), axis=None)

    # construct A & B of null model P (i.e. P_A & P_B)
    #start_time_construct_null_model = time.time()
    #total_degree = 1. / np.sum(d_c, dtype=np.int64)

    start_time_normalized_Q = time.time()
    #dhat_null = np.sqrt(d_c)
    A_of_null_sym = A_of_null * (np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    nor_B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_of_null_sym = B_of_null * nor_B1
    print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))


    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_11_sym =  W_11_sym - A_of_null_sym
    M_12_sym =  W_12_sym - B_of_null_sym
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    pinv_A_new_sym = pinv(M_11_sym)
    M_21_sym = M_12_sym.transpose()

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_sym, R_sym = np.linalg.qr(M_21_sym, mode='reduced')
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_sym = np.dot(R_sym, np.dot(pinv_A_new_sym, R_sym.transpose()))
    S_sym = (S_sym + S_sym.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_sym, U_sym = eigh(S_sym)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

    E_sym = np.real(E_sym)
    ind_sym = np.argsort(E_sym)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U_sym = U_sym[:,ind_sym]
    E_sym = E_sym[ind_sym]
    E_sym = 2 - E_sym
    E_sym = E_sym[:,np.newaxis]
    V_sym = np.dot(Q_sym, U_sym)
    V_sym = V_sym / np.linalg.norm(V_sym, axis = 0)
    print("calculating eigenvalues & -vectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_sym = np.real(V_sym)

    print("symmetric normalized L_{mix}:-- %.3f seconds --" % (time.time() - start_time_symmetric))

    return E_rw,V_rw, E_sym, V_sym, other_data, index




def nystrom_QR_1_signed_sym_rw(raw_data, num_nystrom  = 300, gamma = None): # basic implementation
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
    print('start Nystrom QR decomposition for L_B_sym / L_B_rw (B^+/B^-)') 

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

    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_array = np.concatenate((d1,d2), axis=None)

    # construct null model P 
    start_time_construct_P = time.time()
    total_degree = 1. / np.sum(d_array, dtype=np.int64)
    A_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[0:num_nystrom,np.newaxis].transpose()) 
    B_of_null = total_degree * (d_array[0:num_nystrom,np.newaxis] @ d_array[num_nystrom:num_rows,np.newaxis].transpose())
    print("construct null model P:-- %.3f seconds --" % (time.time() - start_time_construct_P))

    # compute B = W - P
    start_time_construct_B = time.time()
    B_11 = A - A_of_null
    B_12 = B - B_of_null
    print("compute B:-- %.3f seconds --" % (time.time() - start_time_construct_B))

    # positive part of B, i.e. B_{11}^+ & B_{12}^+
    B_11_positive = np.where(B_11 > 0, B_11, 0)   # B_{11}^+
    B_12_positive = np.where(B_12 > 0, B_12, 0)   # B_{12}^+

    # negative part of B, i.e. B_{11}^- & B_{12}^-
    B_11_negative = -np.where(B_11 < 0, B_11, 0)   # B_{11}^-
    B_12_negative = -np.where(B_12 < 0, B_12, 0)   # B_{12}^-


    pinv_A_pos = pinv(B_11_positive)
    B_T_pos = B_12_positive.transpose()
    d1_pos = np.sum(B_11_positive,axis = 1) + np.sum(B_12_positive,axis = 1)
    d2_pos = np.sum(B_T_pos,axis = 1) + np.dot(B_T_pos, np.dot(pinv_A_pos, np.sum(B_12_positive,axis = 1)))
    d_c_pos = np.concatenate((d1_pos,d2_pos),axis = 0)


    pinv_A_neg = pinv(B_11_negative)
    B_T_neg = B_12_negative.transpose()
    d1_neg = np.sum(B_11_negative,axis = 1) + np.sum(B_12_negative,axis = 1)
    d2_neg = np.sum(B_T_neg,axis = 1) + np.dot(B_T_neg, np.dot(pinv_A_neg, np.sum(B_12_negative,axis = 1)))
    d_c_neg = np.concatenate((d1_neg,d2_neg),axis = 0)    


    start_time_symmetric = time.time() 
    # symmetric normalize B^+
    start_time_symmetric_B_pos = time.time()  
    dhat = np.sqrt(1./d_c_pos)
    dhat = np.nan_to_num(dhat)
    #print('degree of B^+: ', dhat)
    B_11_positive_sym = B_11_positive * (np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_12_positive_sym = B_12_positive * B1 
    #B_12_positive = np.nan_to_num(B_12_positive)   
    print("symmetric normalized B^+:-- %.3f seconds --" % (time.time() - start_time_symmetric_B_pos))

    # symmetric normalize B^-
    start_time_symmetric_B_neg = time.time() 
    dhat = np.sqrt(1./d_c_neg)
    dhat = np.nan_to_num(dhat)
    #print('degree of B^-: ', dhat)
    B_11_negative_sym = B_11_negative * (np.dot(dhat[0:num_nystrom,np.newaxis],dhat[0:num_nystrom,np.newaxis].transpose()))
    B1 = np.dot(dhat[0:num_nystrom,np.newaxis], dhat[num_nystrom:num_rows,np.newaxis].transpose())
    B_12_negative_sym = B_12_negative * B1
    #B_12_negative = np.nan_to_num(B_12_negative)    
    print("symmetric normalized B^-:-- %.3f seconds --" % (time.time() - start_time_symmetric_B_neg))

    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_11_sym = B_11_positive_sym - B_11_negative_sym
    M_12_sym =  B_12_positive_sym - B_12_negative_sym
    #print('B: ', B.shape)
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    pinv_A_new = pinv(M_11_sym)
    M_21_sym = M_12_sym.transpose()

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_sym, R_sym = np.linalg.qr(M_21_sym, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_sym = np.dot(R_sym, np.dot(pinv_A_new, R_sym.transpose()))
    #S = R @ pinv_A_new @ R.transpose()
    S_sym = (S_sym + S_sym.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_sym, U_sym = eigh(S_sym)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

    E_sym = np.real(E_sym)
    ind = np.argsort(E_sym)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U_sym = U_sym[:,ind]
    E_sym = E_sym[ind]
    E_sym = 2 - E_sym
    E_sym = E_sym[:,np.newaxis]
    E_sym = E_sym[np.nonzero(E_sym > 0)]
    num_E = E_sym.shape[0]
    #print('E_sym: ', num_E)
    V_sym = np.dot(Q_sym, U_sym)
    V_sym = V_sym / np.linalg.norm(V_sym, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_sym = np.real(V_sym)
    V_sym = V_sym[:, -num_E:]
    #print('V_sym: ', V_sym.shape)

    print("symmetric normalized L_{mix} (B^+ & B^-):-- %.3f seconds --" % (time.time() - start_time_symmetric))


    start_time_random_walk = time.time()
    start_time_normalized_B_pos = time.time()
    # random walk B^+_11 & B^+_12
    #dhat_pos = 1./d_c_pos
    #dhat_pos = np.nan_to_num(dhat_pos)
    #dhat_pos = np.expand_dims(dhat_pos, axis=-1)
    #B_11_positive_rw = B_11_positive * (np.dot(dhat_pos[0:num_nystrom,np.newaxis],dhat_pos[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat_pos[0:num_nystrom,np.newaxis], dhat_pos[num_nystrom:num_rows,np.newaxis].transpose())
    #B_12_positive_rw = B_12_positive * B1
    #B_12_positive_rw = B_12_positive * dhat_pos[num_nystrom:num_rows].transpose()  
    
    dhat_pos = 1./d_c_pos
    #dhat_pos = np.nan_to_num(dhat_pos)
    #id_mat = np.ones(num_nystrom)
    #id_mat = np.expand_dims(id_mat, axis=-1)
    #print('id_mat: ', id_mat.shape)
    #id_mat_other = np.ones(other_rows)
    #id_mat_other = np.expand_dims(id_mat_other, axis=-1)
    B_11_positive_rw =  B_11_positive * dhat_pos[0:num_nystrom,np.newaxis].transpose()
    B_12_positive_rw =  B_12_positive * dhat_pos[num_nystrom:num_rows,np.newaxis].transpose() 
    print("random walk B_11^+ & B_12^+:-- %.3f seconds --" % (time.time() - start_time_normalized_B_pos))

    
    # construct B^-_11 & B^-_12
    start_time_normalized_B_neg = time.time()
    dhat_neg = 1./d_c_neg
    #dhat_neg = np.nan_to_num(dhat_neg)
    #dhat_neg = np.expand_dims(dhat_neg, axis=-1)
    #print('degree of B^-: ', dhat)
    #B_11_negative_rw = B_11_negative * (np.dot(dhat_neg[0:num_nystrom,np.newaxis],dhat_neg[0:num_nystrom,np.newaxis].transpose()))
    #B1 = np.dot(dhat_neg[0:num_nystrom,np.newaxis], dhat_neg[num_nystrom:num_rows,np.newaxis].transpose())
    #B_12_negative_rw = B_12_negative * B1
    B_11_negative_rw = B_11_negative * dhat_neg[0:num_nystrom,np.newaxis].transpose()
    B_12_negative_rw = B_12_negative * dhat_neg[num_nystrom:num_rows,np.newaxis].transpose()
    print("random walk B_11^- & B_12^-:-- %.3f seconds --" % (time.time() - start_time_normalized_B_neg)) 

    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_11_rw = B_11_positive_rw - B_11_negative_rw
    M_12_rw =  B_12_positive_rw - B_12_negative_rw
    #print('B: ', B.shape)
    print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
  
    pinv_A_new = pinv(M_11_rw)
    M_21_rw = M_12_rw.transpose()

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_rw, R_rw = np.linalg.qr(M_21_rw, mode='reduced')
    #print('R: ',R.shape)
    print("QR decomposition of M_{21}:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_rw = np.dot(R_rw, np.dot(pinv_A_new, R_rw.transpose()))
    S_rw = (S_rw + S_rw.transpose())/2.
    print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_rw, U_rw = eigh(S_rw)
    print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

    E_rw = np.real(E_rw)
    ind = np.argsort(E_rw)[::-1]

    # calculating eigenvectors
    start_time_compute_eigenvectors = time.time()
    U_rw = U_rw[:,ind]
    E_rw = E_rw[ind]
    E_rw = 2 - E_rw
    E_rw = E_rw[:,np.newaxis]
    E_rw = E_rw[np.nonzero(E_rw > 0)]
    num_E = E_rw.shape[0]
    #print('E_rw: ', E_rw.shape)
    V_rw = np.dot(Q_rw, U_rw)
    V_rw = V_rw / np.linalg.norm(V_rw, axis = 0)
    print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_rw = np.real(V_rw)
    V_rw = V_rw[:, -num_E:]
    #print('V_rw: ', V_rw.shape)

    print("random walk L_{mix} (B^+ & B^-):-- %.3f seconds --" % (time.time() - start_time_random_walk))

    return E_rw, V_rw, E_sym, V_sym, other_data, index