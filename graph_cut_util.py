from quimb import sparse
import scipy
import scipy.sparse as sp
#import scipy as sp
import numpy as np
import time
from scipy.sparse.linalg import eigsh
from numpy.random import permutation
from scipy.spatial.distance import cdist
from scipy.linalg import pinv
from scipy.linalg import sqrtm  
from scipy.linalg import eigh
from scipy.linalg import eig
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs, eigsh, svds
from graph_mbo.utils import get_initial_state_1

## graph_cut.build_graph
_graph_params_default_values = {'affinity': 'rbf', 'n_neighbors': None, 
'Laplacian_type': 'n','gamma': None, 'Neig' : None, 'Eig_solver': 'full', 
'num_nystrom': None, 'neighbor_type':'full','laplacian_matrix_':None }


def build_affinity_matrix_new(raw_data,gamma=None,affinity='rbf', n_neighbors=10, neighbor_type='knearest'):
#                          Eig_solver='full', Laplacian_type='n'):
    """ Build affinity matrix. Wrappers using sklearn modules
    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    graph_params : Parameters with fields below:

        affinity : string
            'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
            'z-p' : adaptive kernel 
                specify n_neighbors
            '0-1' : return an unweighted graph 
                specify n_neighbors
        gamma : double
            width of the rbf kernel
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. 
        neighbor_type : string. 
            'full' 'knearest'
        Laplacian_type : 'n', normalized, 'u', unnormalized
    Return 
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        affinity matrix

    """ 

    # compute the distance matrix
    affinity_matrix_ = None
    if affinity == 'z-p': #Z-P distance, adaptive RBF kernel, currently slow!!
        if n_neighbors is None:
            raise ValueError("Please Specify number nearest points in n_neighbors")
        k = n_neighbors
        dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')
        tau = np.ones([raw_data.shape[0],1])
        for i, row in enumerate(dist_matrix):
            tau[i] = np.partition(row,k)[k]
        scale = np.dot(tau, tau.T)
        temp = np.exp(-dist_matrix/np.sqrt(scale))
        for i,row in enumerate(temp):
            foo = np.partition(row,row.shape[0]-k-1)[row.shape[0]-k-1]
            row[row<foo] =0
        affinity_matrix_ = np.maximum(temp, temp.T)
    else:
        if neighbor_type != 'full':
            if affinity == '0-1':
                connectivity = kneighbors_graph(raw_data, n_neighbors=n_neighbors, include_self=True)
                affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
                affinity_matrix_[affinity_matrix_ == 1.] = 0. 
                return affinity_matrix_               

                           
            elif neighbor_type == 'knearest':
                distance_matrix = kneighbors_graph(raw_data, n_neighbors=n_neighbors, include_self=True, mode = 'distance')
                distance_matrix = distance_matrix*distance_matrix # square the distance
                dist_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                dist_matrix = np.array(dist_matrix.todense())

        else:   # neighbor_type == 'full'
            dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')

        if affinity == 'rbf':
            # gamma = None
            if gamma is None:
                print("graph kernel width gamma not specified, using default value 1")
                gamma = 1
            else : 
                gamma = gamma
            affinity_matrix_ = np.exp(-gamma*dist_matrix)   # Gaussian function

    affinity_matrix_[affinity_matrix_ == 1.] = 0. 
    d_mean = np.mean(np.sum(affinity_matrix_,axis = 0))
    affinity_matrix_ = affinity_matrix_/d_mean
    return affinity_matrix_




def build_affinity_matrix(raw_data,graph_params):
    """ Build affinity matrix. Wrappers using sklearn modules
    Parameters
    -----------
    raw_data : ndarray, shape (n_samples, n_features)
        Raw input data.
    graph_params : Parameters with fields below:

        affinity : string
            'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
            'z-p' : adaptive kernel 
                specify n_neighbors
            '0-1' : return an unweighted graph 
                specify n_neighbors
        gamma : double
            width of the rbf kernel
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. 
        neighbor_type : string. 
            'full' 'knearest'
        Laplacian_type : 'n', normalized, 'u', unnormalized
    Return 
    ----------
    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        affinity matrix

    """ 

    # compute the distance matrix
    affinity_matrix_ = None
    if graph_params.affinity == 'z-p': #Z-P distance, adaptive RBF kernel, currently slow!!
        if graph_params.n_neighbors is None:
            raise ValueError("Please Specify number nearest points in n_neighbors")
        k = graph_params.n_neighbors
        dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')
        tau = np.ones([raw_data.shape[0],1])
        for i, row in enumerate(dist_matrix):
            tau[i] = np.partition(row,k)[k]
        scale = np.dot(tau, tau.T)
        temp = np.exp(-dist_matrix/np.sqrt(scale))
        for i,row in enumerate(temp):
            foo = np.partition(row,row.shape[0]-k-1)[row.shape[0]-k-1]
            row[row<foo] =0
        affinity_matrix_ = np.maximum(temp, temp.T)
    else:
        if graph_params.neighbor_type != 'full':
            if graph_params.affinity == '0-1':
                connectivity = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True)
                affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
                affinity_matrix_[affinity_matrix_ == 1.] = 0. 
                return affinity_matrix_               

                           
            elif graph_params.neighbor_type == 'knearest':
                distance_matrix = kneighbors_graph(raw_data, n_neighbors=graph_params.n_neighbors, include_self=True, mode = 'distance')
                distance_matrix = distance_matrix*distance_matrix # square the distance
                dist_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                dist_matrix = np.array(dist_matrix.todense())

        else:
            dist_matrix = cdist(raw_data,raw_data,'sqeuclidean')

        if graph_params.affinity == 'rbf':
            gamma = None
            if graph_params.gamma is None:
                print("graph kernel width gamma not specified, using default value 1")
                gamma = 1
            else : 
                gamma = graph_params.gamma
            affinity_matrix_ = np.exp(-gamma*dist_matrix)

    affinity_matrix_[affinity_matrix_ == 1.] = 0. 
    d_mean = np.mean(np.sum(affinity_matrix_,axis = 0))
    affinity_matrix_ = affinity_matrix_/d_mean
    return affinity_matrix_




def affinity_matrix_to_laplacian(W, mode = 'n'):  # output: L_{mix} = L_A + Q_A
    """ Build normalized Laplacian Matrix from affinity matrix W
    For dense Laplaians only. For sparse Laplacians use masks

    Parameters
    -----------
    W : affinity_matrix_

    """
    if sp.issparse(W):
        W = np.array(W.todense())  # currently not handeling sparse matrices separately, converting it to full
   

    if mode == 'n' : # normalized laplacian
    #    n_nodes = W.shape[0]
    #    Lap = -W.copy()
        # set diagonal to zero
    #    Lap.flat[::n_nodes + 1] = 0
    #    d = -Lap.sum(axis=0)
    #    d = np.sqrt(d)
    #    d_zeros = (d == 0)
    #    d[d_zeros] = 1
    #    Lap /= d
    #    Lap /= d[:, np.newaxis]
    #    Lap.flat[::n_nodes + 1] = (1 - d_zeros).astype(Lap.dtype)

        A_absolute_matrix = np.abs(W)
        degree = np.array(np.sum(A_absolute_matrix, axis=-1)).flatten()
        dergee_di_null = np.expand_dims(degree, axis=-1)
        #print('max degree: ',degree.shape)
        #print('degree d_i type: ', dergee_di_null.shape)
        num_nodes = len(degree)

        # compute unsigned laplacian
        degree_diag = np.diag(degree)
        #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
        graph_laplacian = degree_diag - W    # L_A = D - A
        degree_inv = np.diag((1 / degree))
        #degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)   # obtain D^{-1}
        #print('D^{-1}: ', degree_inv.shape)
        nor_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)    # obtain L_A_{sym}

        null_model = np.zeros((len(degree), len(degree)))
        total_degree = np.sum(A_absolute_matrix)
        null_model = (dergee_di_null @ dergee_di_null.transpose())/ total_degree

        degree_null_model = np.array(np.sum(null_model, axis=-1)).flatten()
        #print('degree_null_model type: ',type(degree_null_model))
        #num_nodes_null_model = len(degree_null_model)
        #degree_diag_null_model = sp.sparse.spdiags([degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)
        degree_diag_null_model = np.diag(degree_null_model)   
        signless_laplacian_null_model = degree_diag_null_model + null_model  # Q_P = D + P(null model)
        #signless_degree_inv = sp.sparse.spdiags([1.0 / degree_null_model], [0], num_nodes_null_model, num_nodes_null_model)   # obtain D^{-1}
        signless_degree_inv = np.diag((1.0/degree_null_model))
        #print('D^{-1}: ', degree_inv.shape)
        nor_signless_laplacian = np.sqrt(signless_degree_inv) @ signless_laplacian_null_model @ np.sqrt(signless_degree_inv)
 
        l_mix = nor_signless_laplacian + nor_graph_laplacian

        return l_mix
    if mode == 'u' : # unnormalized laplacian L_{mix}
        n_nodes = W.shape[0]
        Lap = W.copy()
        # set diagonal to zero
        Lap.flat[::n_nodes + 1] = 0
        d = Lap.sum(axis=0)
        Lap = np.diag(d) - Lap
        #signless_lap =  np.diag(d) + Lap 
        #lap_mix = Lap + signless_lap   
        return Lap

def build_laplacian_matrix(raw_data,graph_params):
    """ Wrapper for building the normalized Laplacian directly from raw data

    """
    W  = build_affinity_matrix(raw_data , graph_params)
    if graph_params.Laplacian_type == 'n':
        return affinity_matrix_to_laplacian(W,mode = 'n')
    else:
        return affinity_matrix_to_laplacian(W,mode = 'u')

def generate_eigenvectors(L,Neig):
    """ short hand for using scipy arpack package

    Parameters
    -----------
    L : laplacian_matrix_

    """
    return eigsh(L,Neig,which = 'SA')    



## graph_cut.misc

def imageblocks(im, width, format = 'flat'):
    """Extract all blocks of specified size from an image or list of images
    Automatically pads image to ensure num_blocks = num_pixels
    """

    # See
    # http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
    im = np.pad(im,[[width,width],[width,width],[0,0]] ,mode = 'symmetric' )
    Nr, Nc, n_channels = im.shape
    blksz = 2*width+1
    shape = (Nr-2*width, Nc-2*width, blksz, blksz, n_channels)
    strides = im.itemsize*np.array([Nc*n_channels, n_channels, Nc*n_channels, n_channels, 1])
    sb = np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)
    sb = np.ascontiguousarray(sb)
    sb = sb[:(Nr-2*width),:Nc-2*width,:,:,:]
    sb = np.ascontiguousarray(sb)
    sb.shape = (-1, blksz, blksz, n_channels)
    n_samples = sb.shape[0]
    if format == 'flat':
        return sb.transpose([0,3,1,2]).reshape(n_samples,n_channels*blksz*blksz)
    else:
        return sb



def generate_random_fidelity(ind, perc):
    """  generate perc percent random fidelity out of index set ind

    Parameters
    -----------
    ind : ndarray, (n_sample_in_class, )
    perc : float, percent to sample

    """
    ind = np.array(ind)
    num_sample = int(np.ceil(len(ind)*perc))
    ind2 = np.random.permutation(len(ind))
    return ind[ind2[:num_sample]]


def to_standard_labels(labels):
    """  convert any numeric labeling, i.e., labels that are numbers, to
    standard form, 0,1,2,...

    Parameters
    -----------
    labels : ndarray, (n_labels, )

    Return 
    -----------
    out_labels : ndarray, (n_labels, )
    """    
    tags = np.unique(labels)
    out_labels = np.zeros(labels.shape)
    for i, tag in enumerate(tags):
        out_labels[labels == tag] = i
    return out_labels.astype(int)

def vector_to_labels(V):
    """  convert a multiclass assignment vector (n_samples, n_class) 
    to a standard label 0,1,2... (n_samples,) by projecting onto largest component

    Parameters
    -----------
    V : ndarray, shape(n_samples,n_class)
        class assignment vector for multiclass

    """
    return np.argmax(V, axis = 1)


def labels_to_vector(labels,vec_dim = None):
    """  convert a standard label 0,1,2... (n_samples,)
    to a multiclass assignment vector (n_samples, n_class) by assigning to e_k

    Parameters
    -----------
    labels : ndarray, shape(n_samples,)

    """
    # labels = to_standard_labels(in_labels)
    labels = labels.astype(int)
    if vec_dim is None:
        n_class = np.max(labels)+1
    else:
        n_class = vec_dim
    vec = np.zeros([labels.shape[0], n_class])
    for i in range(n_class):
        vec[labels == i,i] = 1.
    return vec

def standard_to_binary_labels(labels):
    """ convert standard labeling 0,1 to binary labeling -1, 1
    """
    out_labels = np.zeros(labels.shape)
    foo = np.unique(labels)
    out_labels[labels == foo[0]] = -1
    out_labels[labels == foo[1]] = 1 
    return out_labels

def to_binary_labels(labels):
    temp = to_standard_labels(labels)
    return standard_to_binary_labels(temp)

def generate_initial_value_multiclass(opt , n_samples = None, n_class = None): 
    """  generate initial value for multiclass classification. 
    an assignment matrix is returned 

    Parameters
    -----------
    opt: string :{'rd_equal','rd'}
        options for generating values
    n_samples : int
        number of nodes in the graph

    Return
    -------
    u_init : ndarray, shape(n_samples, n_class)

    """   
    
    if opt == 'rd_equal':
        ind = permutation(n_samples)
        u_init = np.zeros((n_samples, n_class)) 
        sample_per_class = n_samples // n_class
        for i in range(n_class):
            u_init[ind[i*sample_per_class:(i+1)*sample_per_class], i] = 1
            
        return u_init
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = [n_samples,n_class])
        u_init = labels_to_vector(vector_to_labels(u_init))
        return u_init

def compute_error_rate(labels, ground_truth):
    """ compute the error rate of a classification given ground_truth and the labels 
    since for clustering the order of the labels are relative, we will automatically 
    match ground_truth with labels according to the highest percentage of population 
    in ground truth 

    Parameters
    -----------
    labels : ndarray, shape(n_samples, )
    ground_truth : ndarray, shape(n_samples, )
    """
    # format the labels
    labels = to_standard_labels(labels).astype(int)
    ground_truth = to_standard_labels(ground_truth).astype(int)
    format_labels = np.zeros(labels.shape).astype(int)
    temp = np.unique(labels)
    for tag in temp:
       format_labels[labels == tag] = np.argmax(np.bincount(ground_truth[labels == tag].astype(int)))
    return float(len(ground_truth[format_labels!= ground_truth]))/float(len(ground_truth))

def generate_initial_value_binary(opt = 'rd_equal', V = None, n_samples = None):
    """  generate initial value for binary classification. 
    individual values are -1, 1 valued. 

    Parameters
    -----------
    opt: string :{'rd_equal','rd','eig'}
        options for generating values
    V: ndarray (n_samples, Neig)
        Eigenvector to generate initial condition. 
    n_samples : int
        number of nodes in the graph

    """    
    res = None
    if opt == 'rd_equal':
        ind = permutation(n_samples)
        u_init = np.zeros(n_samples)
        mid = n_samples//2
        u_init[ind[:mid]] = 1
        u_init[ind[mid:]] = -1
        res =  u_init
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = n_samples)
        res = u_init
    elif opt == 'eig':
        res =  V[:,1].copy()
    return res

class Parameters: # write a parameter class for easier parameter manipulation
    """ Class for managing parameters. Initialize with any keyword arguments
    supports adding, setting, checking, deleting. 



    Methods :
    -----------    
    clear(), isin(), set_parameters(), set_to_default_parameters()

    """ 
    def __init__(self, **kwargs):
        for name in kwargs:
            if type(kwargs[name]) is type({}):
                setattr(self,name,Parameters(**kwargs[name]))
            else:
                setattr(self,name,kwargs[name])
    def clear(self,*args):
        if args:
            for name in args:
                delattr(self, name)
        else: # delete everything 
            fields = self.__dict__.copy()
            for name in fields:
                delattr(self,name)
    def isin(self, *args): #short hand for determining 
        if args:
            for name in args:
                if not hasattr(self, name):
                    return False
            return True
        else:
            return True 
    def set_parameters(self,clear_params = False, **kwargs): #basically just the constructor
        if clear_params:
            self.clear()    
        for name in kwargs:
            if type(kwargs[name]) is type({}):
                setattr(self,name,Parameters(**kwargs[name]))
            else:
                setattr(self,name,kwargs[name])
    def set_to_default_parameters(self, default_values):
        """ complete the missing entries of a set of Parameters
        using the default_values provided
        """
        for name in default_values:
            if not hasattr(self,name):
                setattr(self,name,default_values[name])       


class ImageHandler:
    """ Class for reading and doing patch operations on images 
        ----------> (dim[1] / width / x)
        |
        |   Image
        |
        v
      (dim[0] /height / y)

      flatten direction : dim 1, dim 0
    """
    def __init__(self,image):     
        self.data = image
        self.n_channels = image.shape[2]
        self. width = image.shape[1]
        self. height = image.shape[0]
    def imageblocks(self,half_patch_size,format = 'flat'):
        return imageblocks(self.data, width = half_patch_size, format = format)
    def to_2d(self,u_1d):
        assert(u_1d.shape[0] == self.height * self.width)
        return u_1d.reshape(self.height,self.width)
    def block_to_1dindex(self,x,y,w,h):
        assert(x>=0 and y>=0)
        assert(x+w<=self.width+1 and y+h <= self.height+1 )
        res = np.zeros(w*h)
        for i, x_ind in enumerate(range(x,x+w)):
            for j, y_ind in enumerate(range(y,y+h)):
                res[i*h+j] = y_ind*self.width + x_ind
        return res.astype(int)
    def generate_fidelity_from_block(self,x,y,w,h,value):
        ind = self.block_to_1dindex(x,y,w,h)
        return np.concatenate((ind[::,np.newaxis],value*np.ones([w*h,1])),axis = 1)


## graph_cut.nystrom
def flatten_23(v): # short hand for the swapping axis
    return v.reshape(v.shape[0],-1, order = 'F')

 
def nystrom(raw_data, num_nystrom  = 300, sigma = None): # basic implementation
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
    if sigma is None:
        print("graph kernel width not specified, using default value 1")
        sigma = 1

    num_rows = raw_data.shape[0]
    index = permutation(num_rows)
    if num_nystrom == None:
        raise ValueError("Please Provide the number of sample points in num_nystrom")
    sample_data = raw_data[index[:num_nystrom]]
    other_data = raw_data[index[num_nystrom:]]


    # calculating B
    other_points = num_rows - num_nystrom
    distb = cdist(sample_data,other_data,'sqeuclidean')
    if sigma == None:
        sigma = np.percentile(np.percentile(distb, axis = 1, q = 5),q = 40) # a crude automatic kernel
    B = np.exp(-distb/sigma).astype(np.float32)    

    # calculating A
    dista = cdist(sample_data,sample_data,'sqeuclidean')
    A = np.exp(-dista/sigma).astype(np.float32)
        #A.flat[::A.shape[0]+1] = 0

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
    #E, U = eigsh(R)
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



class BuildGraph(Parameters):
    """ Class for graph construction and computing the graph Laplacian

    keyword arguments : 
    -----------
    Eig_solver : string
        'full' : compute the full Laplacian matrix 
        'nystrom' : specify Neig, num_nystrom(number of samples), gamma
        'arpack' : specify Neig
    (--Not using Nystrom)
        affinity : string
            'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
            'z-p' : adaptive kernel 
                specify n_neighbors
            '0-1' : return an unweighted graph 
                specify n_neighbors
        gamma : double
            width of the rbf kernel
        n_neighbors : integer
            Number of neighbors to use when constructing the affinity matrix using
            the nearest neighbors method. 
        neighbor_type : string. 
            'full' 'knearest'
    (--Using Nystrom)
        affinity : only 'rbf' 
        gamma : required
        num_nystrom : number of sample points
    Laplacian_type : 'n', normalized, 'u', unnormalized

    Methods :
    -----------    
    build_Laplacian(raw_data)
    """ 
    def __init__(self, **kwargs): #interface for specifically setting graph parameters
        self.set_parameters(**kwargs)
        self.set_to_default_parameters(_graph_params_default_values)
        if self.affinity == '0-1':
            self.neighbor_type = 'knearest'


    def build_Laplacian(self,raw_data):
        """ Build graph Laplacian

        Input : 
        -----------
        raw_data : ndarray, shape (n_samples, n_features)
            Raw input data.

        """ 
        self.laplacian_matrix_ = None
        if self.Eig_solver == 'nystrom': # add code for Nystrom Extension separately   
            if self.num_nystrom is None:
                raise ValueError("Please Provide the number of sample points in num_nystrom")
            if self.gamma is None:
                print("Warning : Kernel width gamma not provided. Using Default Estimation in Nystrom")                      
            E,V= nystrom(raw_data = raw_data, num_nystrom  = self.num_nystrom, sigma = self.gamma)
            E = E[:self.Neig]
            V = V[:,:self.Neig]
            self.laplacian_matrix_ = {'V': V, 'E': E}
        else: 
            graph_params = Parameters(**self.__dict__) # for backward compatibility with older version
            Lap = build_laplacian_matrix(raw_data = raw_data,graph_params = graph_params)
            if self.Eig_solver  == 'arpack':
                E,V = generate_eigenvectors(Lap,self.Neig)
                E = E[:,np.newaxis]
                self.laplacian_matrix_ = {'V': V, 'E': E}
                return 
            elif self.Eig_solver == 'full':
                self.laplacian_matrix_ = Lap
                return
            else:
                raise NameError("Eig_Solver Needs to be either 'nystrom', 'arpack' or 'full' ")



## graph_cut.graph_cluster.lap_cluster

def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()

def _diffusion_step_eig_new(v,V,E,dt,inner_step_count):
    """diffusion on graphs
    """
    dti = dt / (2 * inner_step_count)
    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1 + dti * E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1 + dti * E)))
        return u_new.ravel()

def _gl_forward_step(u_old,dt,eps):
    v = u_old-dt/eps*(np.power(u_old,3)-u_old) #double well explicit step
    return v

def _l2_fidelity_gradient_binary(u_old,dt,fid,eta):
    temp = fid[:,1].ravel()
    v = u_old.copy()
    v[fid[:,0].astype(int).ravel()] = v[fid[:,0].astype(int).ravel()]+ dt*eta*(temp-v[fid[:,0].astype(int).ravel()]) # gradient step
    return v    

def _mbo_forward_step_binary(u): #thresholding
    v = u.copy()
    v[v<0] = -1
    v[v>0] = 1
    return v

def _l2_fidelity_gradient_multiclass(u_old,dt,fid_ind,fid_vec,eta):
    # temp = fid[:,1].ravel()
    # temp = util.labels_to_vector(temp) # convert to matrix form
    v = u_old.copy()
    v[fid_ind.astype(int).ravel(),:] = v[fid_ind.astype(int).ravel(),:]+ dt*eta*(fid_vec-v[fid_ind.astype(int).ravel(),:]) # gradient step
    return v    

def _mbo_forward_step_multiclass(u): #thresholding
    return labels_to_vector(vector_to_labels(u),vec_dim = u.shape[1])

def threshold(u, thre_val = 0):
    w = u.copy()
    w[w<thre_val] = 0
    w[w>thre_val] = 1
    return w


################################################################################
####################### Main Classifiers for Classification #####################
################################################################################


##### Binary Ginzburg with fidelity, using eigenvectors #######
def gl_binary_supervised_eig(V,E,fid,dt,u_init,eps = 1,eta = 1, tol = 1e-5,Maxiter = 500):
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eps : scalar, 
        diffuse interface parameter
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    u_init : ndarray, shape (n_samples ,1)
        initial u_0 for the iterations
    """

    #performing the Main GL iteration with fidelity
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = _gl_forward_step(u_old,dt,eps)
        v = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
        u_new = _diffusion_step_eig(v,V,E,eps*dt)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new


##### Binary Laplacian Smoothing, using eigenvectors #######
def lap_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Laplacian Smoothing (Used for Benchmarking. Not actually in the LaplacianClustering class)
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = _l2_fidelity_gradient_binary(u_old,dt,fid = fid, eta = eta)
        u_new = _diffusion_step_eig(w,V,E,dt)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new 



##### Binary MBO with fidelity, using eigenvectors #######
def mbo_binary_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 10): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    print(V.shape)
    i = 0
    u_new = u_init.copy()
    u_diff = 1
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_binary(v,dt,fid = fid, eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_binary(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new 


##### MBO Zero Means, using eigenvectors #######
def mbo_zero_means_eig(V,E,dt,u_init,tol = .5,Maxiter = 500,inner_step_count = 5): # inner stepcount is actually important! and can't be set to 1...
    """ The MBO scheme with a forced zero mean constraint. Valid only for binary classification. 
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = u_old.copy()
        for k in range(inner_step_count): # diffuse and threshold for a while
            v = _diffusion_step_eig(w,V,E,dt)
            w = v-np.mean(v) # force the 0 mean
        u_new = _mbo_forward_step_binary(w)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new

##### MBO Zero Means, using eigenvectors #######
def gl_zero_means_eig(V,E,dt,u_init,eps = 1, tol = 1e-5,Maxiter = 500, inner_step_count = 5): 
    """ The MBO scheme with a forced zero mean constraint. Valid only for binary classification. 
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    i = 0
    u_new = u_init.copy()
    u_diff = 1

    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        w = u_old.copy()
        for k in range(inner_step_count): # diffuse and threshold for a while
            v = _diffusion_step_eig(w,V,E,eps*dt)
            w = v-np.mean(v) # force the 0 mean
        u_new = _gl_forward_step(w,dt,eps)
        u_diff = (abs(u_new-u_old)).sum()

        i = i+1
    return u_new  

##### Multiclass MBO with fidelity, using eigenvectors #######
def mbo_multiclass_supervised_eig(V,E,fid,dt,u_init,eta = 1, tol = .5,Maxiter = 500,inner_step_count = 10): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    u_init : ndarray, shape(n_samples, n_class)
        initial condition of scheme
    dt : float
        stepsize for scheme
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    #print('V shape: ', V.shape)
    i = 0
    u_new = u_init.copy()
    u_diff = 1
    fid_ind = fid[:,0]
    fid_vec = labels_to_vector(fid[:,1])
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        for k in range(inner_step_count):
            w = _l2_fidelity_gradient_multiclass(v,dt,fid_ind = fid_ind, fid_vec = fid_vec,eta = eta)
            v = _diffusion_step_eig(w,V,E,dt)
        u_new = _mbo_forward_step_multiclass(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new    


##### MBO Modularity, Multiclass Next  #######
def mbo_modularity_eig(V,E,dt,u_init,k_weights,gamma = .5, tol = .5,Maxiter = 500,inner_step_count = 5): # inner stepcount is actually important! and can't be set to 1...
    """ Binary Ginzburg Landau with fidelity using eigenvectors
    Parameters
    -----------
    V : ndarray, shape (n_samples, Neig)
        collection of smallest eigenvectors
    E : ndarray, shape (n_samples, 1)
        collection of smallest eigenvalues
    eta : scalar,
        strength of fidelity
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    u_init : ndarray, shape(n_samples, n_class)
        initial condition of scheme
    dt : float
        stepsize for scheme
    tol : scalar, 
        stopping criterion for iteration
    Maxiter : int, 
        maximum number of iterations
    """
    #performing the Main MBO iteration with fidelity
    i = 0
    if len(k_weights.shape) == 1:
        k_weights = k_weights.reshape(len(k_weights),1)
    # convert u_init to standard multiclass form for binary tags
    if (len(u_init.shape)== 1) or (u_init.shape[1] == 1): 
        u_init = labels_to_vector(to_standard_labels(u_init))
    u_new = u_init.copy()
    u_diff = 10
    while (i<Maxiter) and (u_diff > tol):
        u_old = u_new.copy()
        v = u_old.copy()
        w = v.copy()
        for k in range(inner_step_count):
            graph_mean_v = np.dot(k_weights.T,v)/np.sum(k_weights)
            w += 2.*gamma*dt*k_weights*(v-graph_mean_v)
            v = _diffusion_step_eig(w,V,E,dt)
            #v = _diffusion_step_eig_new(w,V,E,dt,inner_step_count)
            #k = k+1
        u_new = _mbo_forward_step_multiclass(v)
        u_diff = (abs(u_new-u_old)).sum()
        i = i+1
    return u_new, i




def mbo_modularity_given_eig(num_communities,eigval,eigvec,dt,u_init,k_weights, tol=1e-5, 
                            inner_step_count=3, gamma=0.5, max_iter=10000): 

    m = len(eigval)
    print('m: ',m)

    eigval = eigval.reshape((m,))
    print('eigenvalue shape: ', eigval.shape)

    num_nodes = len(k_weights)
    #degree_diag = np.diag(deg)
    #degree_diag = sp.sparse.spdiags([deg], [0], num_nodes, num_nodes)
    #degree = np.array(np.sum(adj_matrix, axis=1)).flatten()
    #print(np.max(degree))
    #num_nodes = len(degree)
    
    target_size = [num_nodes // num_communities for i in range(num_communities)]
    target_size[-1] = num_nodes - sum(target_size[:-1])

    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which= "SA",)

    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    u = get_initial_state_1(num_nodes, num_communities, target_size)
    print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))
    
    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u.copy()
    
    start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()
        #a = V_sign.transpose() @ u_old
        #demon = sp.sparse.spdiags([1 / (1 + dt * D)], [0], m, m)

        #start_time_diffusion = time.time()

        demon = sp.spdiags([np.exp(- 0.5 * eigval * dt)],[0],m,m) @ eigvec.transpose()
        #for j in range(inner_step_count):
            # Solve system (apply CG or pseudospectral)
        u_half = eigvec @ (demon @ u_old)  # Project back into normal space

        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        #u_new = apply_threshold(u_half, target_size, thresh_type)
        u_new = _mbo_forward_step_multiclass(u_half)
        
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 

        #start_time_stop_criterion = time.time()
        # Stop criterion

        stop_criterion = (np.abs(u_new - u_old)).sum()
        #stop_criterion = np.linalg.norm(u_new-u_old) / np.linalg.norm(u_new)

        #print("compute stop criterion:-- %.3f seconds --" % (time.time() - start_time_stop_criterion))
        
        n = n+1
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n



def mbo_modularity_inner_step(eigval,eigvec,dt,u_init,k_weights, tol=1e-5,inner_step_count=3,
                        gamma=1, max_iter=10000): # inner stepcount is actually important! and can't be set to 1...
    print('Start with MMBO with finite difference')

    eigval_array = np.ravel(eigval)
    #print('eigenvalue type: ', type(eigval))
    #print('eigevectors type: ', type(eigvec))
    m = eigval.shape[0]
    #start_time_lap_mix = time.time()

    #laplacian_mix = graph_laplacian + signless_laplacian_null_model  # L_{mix} = L_A_{sym} + Q_P
    #print('L_{mix} shape: ',laplacian_mix.shape)
    #print("compute laplacian_mix:-- %.3f seconds --" % (time.time() - start_time_lap_mix))
    
    #start_time_eigendecomposition = time.time()
    # compute eigenvalues and eigenvectors
    #D_sign, V_sign = eigsh(
    #    laplacian_mix,
    #    k=m,
    #    v0=np.ones((laplacian_mix.shape[0], 1)),
    #    which= "SA",)
    #print('D_sign shape: ', D_sign.shape)
    #print('V_sign shape: ', V_sign.shape)
    #print("compute eigendecomposition:-- %.3f seconds --" % (time.time() - start_time_eigendecomposition))
   
    #start_time_initialize = time.time()
    # Initialize parameters
    #u = get_initial_state(
    #    num_nodes,
    #    num_communities,
    #    target_size,
    #    type=initial_state_type,
    #    fidelity_type=fidelity_type,
    #    fidelity_V=fidelity_V,)

    #u = get_initial_state_1(num_nodes, num_communities, target_size)
    #print("compute initialize u:-- %.3f seconds --" % (time.time() - start_time_initialize))


    # Perform MBO scheme
    n = 0
    stop_criterion = 10
    u_new = u_init.copy()
    
    start_time_MBO_iteration = time.time()
    while (n < max_iter) and (stop_criterion > tol):
        u_old = u_new.copy()

        dti = dt / (2 * inner_step_count)

        #start_time_diffusion = time.time()
        #demon = sp.sparse.spdiags([1.0 / (1.0 + dti * D_sign)], [0], m, m) @ V_sign.transpose()
        
        #for j in range(inner_step_count):
            # Solve system (apply CG or pseudospectral)
            #u_half = V_sign @ (demon @ u_old)  # Project back into normal space
                
        # Apply thresholding 
        #u_new = apply_threshold(u_half, target_size, thresh_type)
        #u_new = _mbo_forward_step_multiclass(u_half)

        
        demon = sp.spdiags([1 / (1 + dti * eigval_array)], [0], m, m) @ eigvec.transpose()
        #demon = sp.sparse.spdiags([np.exp(-D*dt)],[0],m,m)
        
        for j in range(inner_step_count):
            u_half = eigvec @ (demon @ u_old)
        
        #print("compute MBO diffusion step:-- %.3f seconds --" % (time.time() - start_time_diffusion))
        
        #start_time_thresholding = time.time()
        # Apply thresholding 
        u_new = _mbo_forward_step_multiclass(u_half)
        #print("compute MBO thresholding:-- %.3f seconds --" % (time.time() - start_time_thresholding)) 
        
        start_time_stop_criterion = time.time()
        
        # Stop criterion

        #stop_criterion = (np.abs(u_new - u_old)).sum()
        stop_criterion = scipy.linalg.norm(u_new-u_old) / scipy.linalg.norm(u_new)


        n = n+1
    print("compute the whole MBO iteration:-- %.3f seconds --" % (time.time() - start_time_MBO_iteration))

    return u_new, n





#######################################################################################
########################## The Main Class Definitions #################################
#######################################################################################

class LaplacianClustering(Parameters):
    """ Apply a Laplacian Graph-cut Solver(either MBO or Ginzburg-Landau) to solve
    a semi-supervised or unsupervised clustering problem. 
    semi-supervised minimizes approximately |u|_GraphTV + (u-f)^2, f being the fidelity 
    unsupervised minimizes approximately |u|_GraphTV + balancing term for cluster size
    currently only supports binary classifiation. 

    Class Overview:
    -----------    
        Attributes:    
        -- various scheme specific parameters(scheme_type, n_class...)
        -- self.graph : A BuildGraph Object.
            Containing graph params and the computed graph Laplacian. 
        -- self.data : A Parameter Object. 
            Containing the raw data, ground truth label 
            
        Methods : 
        -- Constructor : set scheme specific parameters. 
        -- build_graph : build the graph Laplacian, specifying the graph parameters
        -- load_data : load raw data into model.  
        -- generate_random_fidelity : generate some random fidelity
        -- fit_predict : predict labels for the data. 


    Class Constructor Parameters
    ----------------------------
    scheme_type : String {'GL_fidelity','MBO_fidelity','spectral_clustering',
                    GL_zero_means','MBO_zero_means','modularity'}
        Types of scheme for the classifiation. First two are for semi-supervised learning, and 
        last four for unsupervised learning.     
    n_class : integer 
        number of classes (This can be inferred from ground_truth if provided)   
    u_init : ndarray, shape(n_samples,) or (n_samples,k) for multiclass
        initial labels or score for algorithm
    fid : ndarray, shape(num_fidelity, 2)
        index and label of the fidelity points. fid[i,0] 
    eta : scalar
        fidelity strength term
    eps : scalar
        diffuse interface parameter(only for Ginzburg-Landau schemes)
    dt : scalar
        Learning stepsize for the scheme
    inner_step_count : int
        for MBO, GL_zero_means, Modularity. Number of times the scheme performs
        "diffuse" + "forcing" before doing "threshold"

    Data Parameters(in self.data)
    ----------------------------    
    ( use load_data() to change values )
    raw_data : ndarray, (n_samples, n_features)
        raw data for the classification task
    ground_truth : ndarray, (n_samples,).{0...K-1} labels.
        labels corresponding to the raw data  

    Graph Parameters and Data(in self.graph)
    ----------------------------    
    ( use set_graph_params() to change values )
    See self.build_graph() for more details

    Other Class Attributes
    ---------------------- 
    laplacian_matrix_ : array-like, shape (n_samples, n_samples)
        graph laplacian matrix or a dictionary {'V':V,'E':'E'} containing eigenvectors and
        eigenvalues of the laplacian. 
    labels_ :
        Labels or the score of each point    
    """ 

    ## default values for relavant parameters. 
    _params_default_values = {'scheme_type': None, 'n_class': None, 'data': Parameters(), 
    'fid':None,'eps':None, 'eta':None, 'dt': None, 'u_init' : None,'inner_step_count' : None,
    'gamma': None} 

    def __init__(self, **kwargs): # look how clean the constructor is using inheritance! 
        self.set_parameters(**kwargs)
        self.set_to_default_parameters(self._params_default_values)
        self.graph = BuildGraph()

    def load_data(self, raw_data  = None, ground_truth = None):
        """
            raw_data : ndarray, (n_samples, n_features)
            ground_truth : ndarray, (n_samples,).{0...K-1} labels.
        """
        if not raw_data is None:
            self.nclass = None # reset number of classes
            self.graph = BuildGraph() # reset the graph every time new data is loaded 
            self.data.raw_data = raw_data
            if hasattr(self,'fid'):
                self.fid = None
        if not ground_truth is None:
            # infer the label from ground_truth if available. 
            self.nclass = None #reset the number of classes
            self.n_class = np.unique(ground_truth).shape[0] 
            if np.unique(ground_truth).shape[0] == 2 :# convert labels binary case
                self.data.ground_truth = to_binary_labels(ground_truth)
            else:
                self.data.ground_truth = to_standard_labels(ground_truth)
            if hasattr(self,'fid'): #reset fidelity after loading ground_truth
                self.fid = None       

    def set_graph_params(self,**kwargs):
        """ Available Parameters for Graphs

        -----------
        Eig_solver : string
            'full' : compute the full Laplacian matrix 
            'nystrom' : specify Neig, num_nystrom(number of samples), gamma
            'arpack' : specify Neig
        (--Not using Nystrom)
            affinity : string
                'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                    specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
                'z-p' : adaptive kernel 
                    specify n_neighbors
                '0-1' : return an unweighted graph 
                    specify n_neighbors
            gamma : double
                width of the rbf kernel
            n_neighbors : integer
                Number of neighbors to use when constructing the affinity matrix using
                the nearest neighbors method. 
            neighbor_type : string. 
                'full' 'knearest'
        (--Using Nystrom)
            affinity : only 'rbf' 
            gamma : required
            num_nystrom : number of sample points
        Laplacian_type : 'n', normalized, 'u', unnormalized
        """

        try : 
            self.graph.set_parameters(**kwargs)
            self.graph.build_Laplacian(self.data.raw_data)
        except : 
            raise AttributeError("self.graph Non-existent. Use .build_Laplacian() to construct the graph object")

    def build_graph(self, **kwargs): # build the graph Laplacian
        """ Construct and compute the graph Laplacian

        keyword arguments : 
        -----------
        Eig_solver : string
            'full' : compute the full Laplacian matrix 
            'nystrom' : specify Neig, num_nystrom(number of samples), gamma
            'arpack' : specify Neig
        (--Not using Nystrom)
            affinity : string
                'rbf' : use rbf kernel  exp(|x-y|^2/gamma)
                    specify gamma, neighbor_type = 'full', or 'knearest' with n_neighbors
                'z-p' : adaptive kernel 
                    specify n_neighbors
                '0-1' : return an unweighted graph 
                    specify n_neighbors
            gamma : double
                width of the rbf kernel
            n_neighbors : integer
                Number of neighbors to use when constructing the affinity matrix using
                the nearest neighbors method. 
            neighbor_type : string. 
                'full' 'knearest'
        (--Using Nystrom)
            affinity : only 'rbf' 
            gamma : required
            num_nystrom : number of sample points
        Laplacian_type : 'n', normalized, 'u', unnormalized
        """
        if kwargs:
            if hasattr(self,'graph'):
                self.clear('graph')
            self.graph = BuildGraph(**kwargs)
        self.graph.build_Laplacian(self.data.raw_data)

    def generate_initial_value(self, opt = 'rd_equal'):
        # infer the label from ground_truth if available. 
        try : 
            if self.n_class is None:
                if not self.data.ground_truth is None:
                    self.n_class = np.unique(self.data.ground_truth).shape[0] 
                else:
                    raise AttributeError("No ground truth found. Need to specify n_class via set_parameters() ")
        except : 
            raise AttributeError("Either the data or the ground_truth is not specified. Cannot infer class number")
        if self.n_class == 2:
            if opt != 'eig':
                self.u_init = generate_initial_value_binary(opt = opt, V = None, n_samples = self.data.raw_data.shape[0])
                if 'modularity' in self.scheme_type: #the modularity method has inherently 0-1 vector labels
                    self.u_init = labels_to_vector(to_standard_labels(threshold(self.u_init)))
            else:
                try:
                    self.u_init = generate_initial_value_binary(opt = 'eig', V = self.graph.laplacian_matrix_['V'])
                except KeyError:
                    raise KeyError("laplacian_matrix_ needs to be in eigenvector format")
        elif opt != 'eig':
            self.u_init = generate_initial_value_multiclass(opt = opt, n_samples = self.data.raw_data.shape[0], n_class = self.n_class)
        else:
            raise NameError("Eig Option is currently unavailable for multiclass data")


    def generate_random_fidelity(self,percent = .05):
        try : 
            tags = np.unique(self.data.ground_truth)
        except AttributeError:
            raise AttributeError("Please provide ground truth")
        self.fid = np.zeros([0,2])
        for i, tag in enumerate(tags):
            ind_temp = generate_random_fidelity(ind =  np.where(self.data.ground_truth == tag)[0] , perc = percent)
            ind_temp = ind_temp.reshape(len(ind_temp), 1)
            tag_temp = tag*np.ones([len(ind_temp),1])
            fid_temp = np.concatenate((ind_temp, tag_temp), axis = 1)
            self.fid = np.concatenate((self.fid,fid_temp), axis = 0)


    def fit_predict(self):
        # build the laplacian if there is non existent
        if hasattr(self,'labels_'):
            self.clear('labels_')
        try:
            if self.graph.laplacian_matrix_ is None:
                raise AttributeError("Build The Graph using build_graph() before calling fit_predict")
        except AttributeError:
            raise AttributeError("Build The Graph using build_graph() before calling fit_predict")
        # infer the label from ground_truth if available. 
        try : 
            if self.n_class is None:
                if not self.data.ground_truth is None:
                    self.n_class = np.unique(self.data.ground_truth).shape[0] 
                else:
                    raise AttributeError("No ground truth found. Need to specify n_class via set_parameters() ")
        except : 
            raise AttributeError("Either the data or the ground_truth is not specified. Cannot infer class number")

        if self.scheme_type.find("fidelity") != -1:
            if self.fid is None:
                print("Fidelity point not provided. Generating 5 percent random fidelity.")
                self.generate_random_fidelity()

        if self.u_init is None:
            print("u_init not provided. Generating random initial condition.")
            self.generate_initial_value()
        if (len(self.u_init.shape) == 1):
            if(self.n_class !=2)  :
                print("u_init dimension not matching n_class. Possiblly out of date. Generating new one")
                self.generate_initial_value()
        elif (self.u_init.shape[1] != self.n_class) :
            print("u_init dimension not matching n_class. Possiblly out of date. Generating new one")
            self.generate_initial_value()

        # check if the laplacian is in eigenvector form
        if type(self.graph.laplacian_matrix_) is dict:
            V = self.graph.laplacian_matrix_['V']
            E = self.graph.laplacian_matrix_['E']
            #Neig = V.shape[1]

        # wrapper to check which scheme to use.     
        if self.scheme_type == 'GL_fidelity':
            inner_step_count = None
            if self.eta is None:
                self.eta = 1
                print("Warning, fidelity strength eta not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = .5
                print("Warning, stepsize dt not supplied. Using default value .1")   
            if self.eps is None:
                self.eps = 1
                print("Warning, scale interface parameter eps not supplied. Using default value 1")                     
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2 :
                    labels = gl_binary_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid ,dt = self.dt, u_init = self.u_init ,
                        eps = self.eps ,eta = self.eta)
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels 
                else:
                    raise ValueError("Ginzburg-Landau Schemes only for 2 class segmentation")
                    return
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")
        elif self.scheme_type == 'MBO_fidelity':   
            inner_step_count = None     
            if self.eta is None:
                self.eta = 1
                print("Warning, fidelity strength eta not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = 1
                print("Warning, stepsize dt not supplied. Using default value 1")                
            if self.inner_step_count is None:
                inner_step_count = 10 # default value
            else : 
                inner_step_count = self.inner_step_count
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_binary_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid 
                        ,dt = self.dt, u_init = self.u_init ,eta = self.eta, inner_step_count = inner_step_count)
                else:
                    res = mbo_multiclass_supervised_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],fid = self.fid ,dt = self.dt, 
                        u_init = self.u_init ,eta = self.eta) 
                    self.labels_ = vector_to_labels(res)                  
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")    
        elif self.scheme_type == 'MBO_zero_means':    
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")   
            if self.inner_step_count is None:
                inner_step_count = 5 # default value
            else : 
                inner_step_count = self.inner_step_count                
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    self.labels_ = mbo_zero_means_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],dt = self.dt
                        , u_init = self.u_init, inner_step_count = inner_step_count)                 
                else:
                    raise ValueError("MBO_zero_means Schemes only for 2 class segmentation")              
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph") 
        elif self.scheme_type == 'GL_zero_means': 
            inner_step_count = None  
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")       
            if self.inner_step_count is None:
                inner_step_count = 10 # default value
            else : 
                inner_step_count = self.inner_step_count  
            if type(self.graph.laplacian_matrix_) is dict:
                if self.n_class == 2:
                    labels = gl_zero_means_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],eps = self.eps,dt = self.dt
                        , u_init = self.u_init , inner_step_count = inner_step_count)
                    self.soft_labels_ = labels
                    labels[labels<0] = -1
                    labels[labels>0] = 1
                    self.labels_ = labels                     
                else:
                    raise ValueError("Ginzburg-Landau Schemes only for 2 class segmentation")              
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph") 

        elif self.scheme_type == 'MBO_modularity':
            inner_step_count = None
            temp = np.ones((self.data.raw_data.shape[0],1))
            print('degree: ', temp.shape)
            if self.inner_step_count is None:
                inner_step_count = 3 # default value
            else : 
                inner_step_count = self.inner_step_count             
            if self.gamma is None:
                self.gamma = 1
                print("Warning, Modularity parameter gamma not supplied. Using default value 1")      
            if self.dt is None:
                self.dt = .1
                print("Warning, stepsize dt not supplied. Using default value .1")                   
            if type(self.graph.laplacian_matrix_) is dict:
                num_communities = self.n_class 
                #res, num_iteration = mbo_modularity_inner_step(self.graph.laplacian_matrix_['E'], self.graph.laplacian_matrix_['V'], 
                #        dt = self.dt, u_init = self.u_init, k_weights = temp, gamma = self.gamma)
                #res, num_iteration = mbo_modularity_given_eig(num_communities, self.graph.laplacian_matrix_['E'], self.graph.laplacian_matrix_['V'], 
                #        dt = self.dt, u_init = self.u_init, k_weights = temp, gamma = self.gamma)
                res, num_iteration = mbo_modularity_eig(self.graph.laplacian_matrix_['V'],self.graph.laplacian_matrix_['E'],k_weights = temp, 
                    dt = self.dt, u_init = self.u_init, gamma = self.gamma ,inner_step_count = inner_step_count)
                print('number of interation: ',num_iteration)
                self.labels_ = vector_to_labels(res)            
            elif type(self.graph.laplacian_matrix_) is np.ndarray:
                print("Full Laplacian Scheme not implemented yet") # use the full Laplacian to solve. Not implemented yet. 
                return
            else : 
                raise AttributeError("laplacian_matrix_ type error. Please rebuild the graph")

        elif self.scheme_type == 'spectral_clustering': # added for benchmark  
        # this just performs k-means after doing spectral projection                
            if type(self.graph.laplacian_matrix_) is dict:
                cf = KMeans(n_clusters = self.n_class)
                temp = self.graph.laplacian_matrix_['V'][:,1:]
                self.labels_ = cf.fit_predict(temp)              
            else : 
                raise AttributeError("Spectral Clustering only supported for laplacian_matrix_ in eigenvector format")

    def compute_error_rate(self):
        try : 
            if (self.data.ground_truth is None):
                raise ValueError("Please provide ground truth labels when using compute_error_rate() ")
        except : 
            raise ValueError("Please provide ground truth labels when using compute_error_rate() ")       
        self.error_rate_ = compute_error_rate(ground_truth = self.data.ground_truth, labels = self.labels_)
        return self.error_rate_

