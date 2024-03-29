
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


def nystrom_QR_l_sym(raw_data, num_nystrom  = 300, tau = None): # basic implementation
    print('raw_data',raw_data.shape)
    print('Start Nystrom extension using QR decomposition for L_sym / L_rw')    

    if tau is None:
        print("graph kernel width not specified, using default value 1")
        tau = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    order_raw_data = raw_data[index]


    # calculating the first k--th column of W
    start_time_calculating_the_first_k_columns_W = time.time()
    first_k_columns_W = rbf_kernel(order_raw_data, sample_data, gamma=tau)
    #distb = cdist(order_raw_data,sample_data,'sqeuclidean')
    #first_k_columns_W = np.exp(-distb/tau).astype(np.float32) 

    # calculating W_11
    start_time_calculating_W_11 = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=tau)
    #dista = cdist(sample_data,sample_data,'sqeuclidean')
    #A = np.exp(-dista/tau).astype(np.float32)
    #print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_W_11))

    # computing the approximation of W_11 & W_21
    start_time_approximation_W21 = time.time()
    pinv_A = pinv(A)
    #print('pinv_A', pinv_A)

    first_k_columns_W_T = first_k_columns_W.transpose()
    d_c = np.dot(first_k_columns_W, np.dot(pinv_A, np.sum(first_k_columns_W_T,axis = 1)))
    d_inverse = np.sqrt(1./d_c)
    d_inverse = np.nan_to_num(d_inverse)
    d_inverse = np.expand_dims(d_inverse, axis=-1)
    first_k_columns_W = first_k_columns_W * d_inverse
    #print("computing the approximation of W_21:-- %.3f seconds --" % (time.time() - start_time_approximation_W21))
    
    # QR decomposition for the approximation of W_21
    start_time_QR_decomposition_approximation_W21 = time.time()
    Q, R = np.linalg.qr(first_k_columns_W)
    #print('R', R)
    #print("QR decomposition for the approximation of W_21:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_W21))
    
    # construct S
    start_time_construct_S = time.time()    
    S = np.dot(R, np.dot(pinv_A, R.transpose()))
    S = (S+S.transpose())/2.
    #print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E, U = eigh(S)
    #print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))
    
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
    time_eig_l_sym = time.time() - start_time_approximation_W21
    #print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_calculating_W_21))
    V = np.real(V)

    start_time_compute_eigenvector_rw = time.time()
    rw_left_eigvec = V * d_inverse
    time_eig_l_rw = time.time() - start_time_approximation_W21
    #print("calculating rw eigenvectors:-- %.3f seconds --" % (time.time() - start_time_calculating_W_21))

    return E, V, rw_left_eigvec, order_raw_data, index, time_eig_l_sym, time_eig_l_rw




def nystrom_QR_l_mix_sym_rw(raw_data, num_nystrom  = 300, tau = None): # basic implementation

    print('Start Nystrom extension using QR decomposition for L_mix_sym / L_mix_rw')     

    if tau is None:
        print("graph kernel width not specified, using default value 1")
        tau = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    order_raw_data = raw_data[index]

    # calculating the first k-th column of W
    start_time_calculating_the_first_k_columns_W = time.time()
    first_k_columns_W = rbf_kernel(order_raw_data, sample_data, gamma=tau)

    # calculating W_21
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=tau)
    #print("calculating W_21:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating W_11
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=tau)
    #print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    
    pinv_A = pinv(A)
    first_k_columns_W_T = first_k_columns_W.transpose()
    d_c = np.dot(first_k_columns_W, np.dot(pinv_A, np.sum(first_k_columns_W_T,axis = 1)))
    
    d_inverse = 1./d_c
    d_inverse = np.expand_dims(d_inverse, axis=-1)

    
    # Construct the null model P
    total_degree = 1. / np.sum(d_c, dtype=np.int64)
    P_first_k_columns = total_degree * d_c[:,np.newaxis] @ d_c[0:num_nystrom,np.newaxis].transpose()

    start_time_random_walk = time.time()
    # random walk W_11 & W_12
    start_time_normalized_W = time.time()
    first_columns_W_rw = first_k_columns_W * d_inverse
    #print("random walk W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    start_time_normalized_Q = time.time()
    first_k_columns_P_rw = P_first_k_columns * d_inverse
    #print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))
    
    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_first_k_column_rw = first_columns_W_rw - first_k_columns_P_rw
    #print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    M_11_rw = M_first_k_column_rw[:num_nystrom, :]
    pinv_A_new_rw = pinv(M_11_rw)

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_rw, R_rw = np.linalg.qr(M_first_k_column_rw, mode='reduced')
    #print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_rw = np.dot(R_rw, np.dot(pinv_A_new_rw, R_rw.transpose()))
    S_rw = (S_rw + S_rw.transpose())/2.
    #print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_rw, U_rw = eigh(S_rw)
    #print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

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
    #print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_rw = np.real(V_rw)

    time_eig_l_mix_rw = time.time() - start_time_random_walk
    #print("random walk L_{mix}:-- %.3f seconds --" % (time.time() - start_time_random_walk))

    start_time_symmetric = time.time()
    # symmetric normalize A and B of W 
    start_time_normalized_W = time.time()
    dhat = np.sqrt(1./d_c)
    dhat = np.expand_dims(dhat, axis=-1)
    first_columns_W_sym = first_k_columns_W * np.dot(dhat, dhat[0:num_nystrom].transpose()) 
    #print("normalized W_11 & W_12:-- %.3f seconds --" % (time.time() - start_time_normalized_W))


    # construct A & B of null model P (i.e. P_A & P_B)
    start_time_normalized_Q = time.time()
    first_k_columns_P_sym = P_first_k_columns * np.dot(dhat, dhat[0:num_nystrom].transpose())
    #print("normalized P_11 & P_12:-- %.3f seconds --" % (time.time() - start_time_normalized_Q))

    
    # computing the approximation of B
    #start_time_approximation_B = time.time()
    M_sym_first_k_column = first_columns_W_sym - first_k_columns_P_sym
    M_11_sym = M_sym_first_k_column[:num_nystrom, :]
    pinv_A_new_sym = pinv(M_11_sym)

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_sym, R_sym = np.linalg.qr(M_sym_first_k_column, mode='reduced')
    #print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_sym = np.dot(R_sym, np.dot(pinv_A_new_sym, R_sym.transpose()))
    S_sym = (S_sym + S_sym.transpose())/2.
    #print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_sym, U_sym = eigh(S_sym)
    #print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

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
    #print("calculating eigenvalues & -vectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_sym = np.real(V_sym)
    time_eig_l_mix_sym = time.time() - start_time_symmetric
    #print("symmetric normalized L_{mix}:-- %.3f seconds --" % (time.time() - start_time_symmetric))

    return E_sym, V_sym, E_rw, V_rw, order_raw_data, index, time_eig_l_mix_sym, time_eig_l_mix_rw




def nystrom_QR_l_mix_B_sym_rw(raw_data, num_nystrom  = 300, tau = None): # basic implementation

    print('Start Nystrom extension using QR decomposition for L_B_sym / L_B_rw (B^+/B^-)') 

    if tau is None:
        print("graph kernel width not specified, using default value 1")
        tau = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    order_raw_data = raw_data[index]

    # calculating the first k-th columns of W
    #start_time_calculating_the_first_k_columns_W = time.time()
    first_k_columns_W = rbf_kernel(order_raw_data, sample_data, gamma=tau)

    # calculating W_21
    start_time_calculating_B = time.time()
    B = rbf_kernel(sample_data, other_data, gamma=tau)
    #print("calculating W_21:-- %.3f seconds --" % (time.time() - start_time_calculating_B))

    # calculating W_11
    start_time_calculating_A = time.time()
    A = rbf_kernel(sample_data, sample_data, gamma=tau)
    #print("calculating W_11:-- %.3f seconds --" % (time.time() - start_time_calculating_A))

    pinv_A = pinv(A)
    first_k_columns_W_T = first_k_columns_W.transpose()
    d_c = np.dot(first_k_columns_W, np.dot(pinv_A, np.sum(first_k_columns_W_T,axis = 1)))

    # construct null model P 
    start_time_construct_P = time.time()
    total_degree = 1. / np.sum(d_c, dtype=np.int64)
    P_first_k_columns = total_degree * d_c[:,np.newaxis] @ d_c[0:num_nystrom,np.newaxis].transpose()
    #print("construct null model P:-- %.3f seconds --" % (time.time() - start_time_construct_P))

    # compute B = W - P
    start_time_construct_B = time.time()
    first_k_columns_B = first_k_columns_W - P_first_k_columns
    #print("compute B:-- %.3f seconds --" % (time.time() - start_time_construct_B))

    # positive part of B, i.e. B_{11}^+ & B_{12}^+
    B_positive = np.where(first_k_columns_B > 0, first_k_columns_B, 0)
    B_11_positive = B_positive[:num_nystrom,:]
    pinv_A_pos = pinv(B_11_positive)

    # negative part of B, i.e. B_{11}^- & B_{12}^-
    B_negative = -np.where(first_k_columns_B < 0, first_k_columns_B, 0)
    B_11_negative = B_negative[:num_nystrom,:]
    pinv_A_neg = pinv(B_11_negative)


    start_time_symmetric = time.time() 
    # symmetric normalize B^+
    #start_time_symmetric_B_pos = time.time()  
    B_positive_T = B_positive.transpose()
    d_c_pos = np.dot(B_positive, np.dot(pinv_A_pos, np.sum(B_positive_T,axis = 1)))
    dhat_pos = np.sqrt(1./d_c_pos)
    dhat_pos = np.nan_to_num(dhat_pos)
    dhat_pos = np.expand_dims(dhat_pos, axis=-1)
    first_columns_B_pos_sym = B_positive * np.dot(dhat_pos, dhat_pos[0:num_nystrom].transpose())
    #print("symmetric normalized B^+:-- %.3f seconds --" % (time.time() - start_time_symmetric_B_pos))

    # symmetric normalize B^-
    #start_time_symmetric_B_neg = time.time() 
    B_negaive_T = B_negative.transpose()
    d_c_neg = np.dot(B_negative, np.dot(pinv_A_neg, np.sum(B_negaive_T,axis = 1)))
    dhat_neg = np.sqrt(1./d_c_neg)
    dhat_neg = np.nan_to_num(dhat_neg)
    dhat_neg = np.expand_dims(dhat_neg, axis=-1)
    first_columns_B_neg_sym = B_negative * np.dot(dhat_neg, dhat_neg[0:num_nystrom].transpose())   
    #print("symmetric normalized B^-:-- %.3f seconds --" % (time.time() - start_time_symmetric_B_neg))

    
    # compute M_{FH} = normalized W - normalized P
    start_time_construct_B = time.time()
    M_sym_first_k_columns = first_columns_B_pos_sym - first_columns_B_neg_sym
    #print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
    
    # computing the approximation of B
    M_11_sym = M_sym_first_k_columns[:num_nystrom, :]
    pinv_A_new = pinv(M_11_sym)


    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_sym, R_sym = np.linalg.qr(M_sym_first_k_columns, mode='reduced')
    #print("QR decomposition of B:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_sym = np.dot(R_sym, np.dot(pinv_A_new, R_sym.transpose()))
    S_sym = (S_sym + S_sym.transpose())/2.
    #print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_sym, U_sym = eigh(S_sym)
    #print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

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
    V_sym = np.dot(Q_sym, U_sym)
    V_sym = V_sym / np.linalg.norm(V_sym, axis = 0)
    #print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_sym = np.real(V_sym)
    V_sym = V_sym[:, -num_E:]
    time_eig_B_sym = time.time() - start_time_calculating_B
    #print("symmetric normalized L_{mix} (B^+ & B^-):-- %.3f seconds --" % (time.time() - start_time_symmetric))


    start_time_random_walk = time.time()
    start_time_normalized_B_pos = time.time()
    # random walk B^+_11 & B^+_12
    dhat_pos = 1./d_c_pos
    dhat_pos = np.nan_to_num(dhat_pos)
    dhat_pos = np.expand_dims(dhat_pos, axis=-1)
    B_positive_rw = B_positive * dhat_pos
    #print("random walk B_11^+ & B_12^+:-- %.3f seconds --" % (time.time() - start_time_normalized_B_pos))

    
    # construct B^-_11 & B^-_12
    start_time_normalized_B_neg = time.time()
    dhat_neg = 1./d_c_neg
    dhat_neg = np.nan_to_num(dhat_neg)
    dhat_neg = np.expand_dims(dhat_neg, axis=-1)
    B_negative_rw = B_negative * dhat_neg
    #print("random walk B_11^- & B_12^-:-- %.3f seconds --" % (time.time() - start_time_normalized_B_neg)) 

    # compute M_{FH} = normalized W - normalized P
    #start_time_construct_B = time.time()
    M_rw_first_k_column = B_positive_rw - B_negative_rw
    #print("compute M_{FH}:-- %.3f seconds --" % (time.time() - start_time_construct_B))
  
    M_11_rw = M_rw_first_k_column[:num_nystrom,:]
    pinv_A_new = pinv(M_11_rw)

    # QR decomposition of B
    start_time_QR_decomposition_approximation_B = time.time()
    Q_rw, R_rw = np.linalg.qr(M_rw_first_k_column, mode='reduced')
    #print("QR decomposition of M_{21}:-- %.3f seconds --" % (time.time() - start_time_QR_decomposition_approximation_B))
    
    # construct S
    start_time_construct_S = time.time()  
    S_rw = np.dot(R_rw, np.dot(pinv_A_new, R_rw.transpose()))
    S_rw = (S_rw + S_rw.transpose())/2.
    #print("construct S:-- %.3f seconds --" % (time.time() - start_time_construct_S))
    
    # do orthogonalization and eigen-decomposition of S
    start_time_eigendecomposition_S = time.time()
    E_rw, U_rw = eigh(S_rw)
    #print("do eigen-decomposition of S:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition_S))

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
    V_rw = np.dot(Q_rw, U_rw)
    V_rw = V_rw / np.linalg.norm(V_rw, axis = 0)
    #print("calculating eigenvectors:-- %.3f seconds --" % (time.time() - start_time_compute_eigenvectors))
    V_rw = np.real(V_rw)
    V_rw = V_rw[:, -num_E:]

    time_eig_B_rw = time.time() - start_time_random_walk
    #print("random walk L_{mix} (B^+ & B^-):-- %.3f seconds --" % (time.time() - start_time_random_walk))

    return E_sym, V_sym, E_rw, V_rw, order_raw_data, index, time_eig_B_sym, time_eig_B_rw




def nystrom_original(raw_data, num_nystrom  = 300, tau = None): # basic implementation
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
    if tau is None:
        print("graph kernel width not specified, using default value 1")
        tau = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]
    order_raw_data = raw_data[index]


    # calculating B
    other_points = num_rows - num_nystrom
    distb = cdist(sample_data,other_data,'sqeuclidean')
    B = np.exp(-distb/tau).astype(np.float32)    

    # calculating A
    dista = cdist(sample_data,sample_data,'sqeuclidean')
    A = np.exp(-dista/tau).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

    # normalize A and B
    pinv_A = pinv(A)
    B_T = B.transpose()
    d1 = np.sum(A,axis = 1) + np.sum(B,axis = 1)
    d2 = np.sum(B_T,axis = 1) + np.dot(B_T, np.dot(pinv_A, np.sum(B,axis = 1)))
    d_c = np.concatenate((d1,d2),axis = 0)
    d_inverse = 1./d_c
    d_inverse = np.nan_to_num(d_inverse)
    dhat = np.sqrt(d_inverse)
    #dhat = np.nan_to_num(dhat)
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
    return E,V, order_raw_data
