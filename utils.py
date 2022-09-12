import numpy as np
from numpy.random import permutation
from sklearn import metrics
from itertools import product


def labels_to_vector(labels,vec_dim = None):
    """  convert a standard label 0,1,2... (n_samples,)
    to a multiclass assignment vector (n_samples, n_class) by assigning to {(1, -1, -1,...), (-1, 1, -1,...)}

    Parameters
    -----------
    labels : ndarray, shape(n_samples,)

    """

    labels = labels.astype(int)
    if vec_dim is None:
        n_class = np.max(labels)+1
    else:
        n_class = vec_dim
        # use TV, then output u contains {0,1}
    vec = np.zeros([labels.shape[0], n_class])
        
        # use signless TV, then output u contains {-1,1}
    #vec = -np.ones([labels.shape[0], n_class])
    for i in range(n_class):
        vec[labels == i,i] = 1.
        
        # output u contains {-1,1}
    vec = np.where(vec > 0, vec, -1)
    return vec



def vector_to_labels(V):
    """  convert a multiclass assignment vector (n_samples, n_class) 
    to a standard label 0,1,2... (n_samples,) by projecting onto largest component

    Parameters
    -----------
    V : ndarray, shape(n_samples,n_class)
        class assignment vector for multiclass

    """
    return np.argmax(V, axis = 1)




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
            # original: only use TV, then u_init is zero matrix
        u_init = np.zeros((n_samples, n_class)) 
            # use signless TV, then u_init = (-1)matrix
        #u_init = -np.ones((n_samples, n_class)) 
        sample_per_class = n_samples // n_class
        for i in range(n_class):
            u_init[ind[i*sample_per_class:(i+1)*sample_per_class], i] = 1
        u_init = np.where(u_init > 0, u_init, -1)
        return u_init
    
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = [n_samples,n_class])
        u_init = labels_to_vector(vector_to_labels(u_init))
        u_init = np.where(u_init > 0, u_init, -1)
        return u_init



def _mbo_forward_step_multiclass(u):   # Thresholding step in the MBO scheme
    return labels_to_vector(vector_to_labels(u),vec_dim = u.shape[1])



def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
    if len(v.shape) > 1:
        value_plus = 1+dt*E
        value_plus = np.expand_dims(value_plus, axis=-1)  # Add an extra dimension in the last axis.
        return np.dot(V,np.divide(np.dot(V.T,v),value_plus))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()



"""Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def inverse_purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) 



def get_modularity_ER(adjacency_mat, community_dict):
#    '''
#    Calculate the modularity. 
#    .. math:: Q = \frac{1}{2m}\sum_{i,j} \(A_ij - \frac{2m}{N(N-1)}\) * \detal_(c_i, c_j)

#    Parameters
#    ----------
#    adjacency_mat : sp.sparse._csr.csr_matrix or ny.ndarray
#        The adjacency matrix of network/graph
#    community_dict : dict
#        A dictionary to store the membership of each node
#        Key is node and value is community index

#    Returns
#    -------
#    float
#        The modularity of `network` given `community_dict`
#    '''

    Q = 0
    num_nodes = adjacency_mat.shape[0]
    list_node = list(range(0, num_nodes))
    total_degree = np.sum(adjacency_mat)  
    N_square = num_nodes * (num_nodes -1)

    Q = np.sum([adjacency_mat[i,j] - total_degree *\
                         1/N_square\
                 for i, j in product(range(len(list_node)),\
                                     range(len(list_node))) \
                if community_dict[list_node[i]] == community_dict[list_node[j]]])
    return Q / total_degree



def label_to_dict(u_label):
    len_label = []
    for i in range(len(u_label)):
        len_label.append(i)
    u_dict = dict(zip(len_label, u_label))
    return u_dict