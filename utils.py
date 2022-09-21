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
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
        #value_plus = 1+dt*E
        #value_plus = np.expand_dims(value_plus, axis=-1)  # Add an extra dimension in the last axis.
        #return np.dot(V,np.divide(np.dot(V.T,v),value_plus))
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



def get_modularity_ER(adjacency_mat, community_dict, gamma=1):
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

    Q = np.sum([adjacency_mat[i,j] - gamma * total_degree *\
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


def dict_to_list_set(dictionary):
    dict_value_list = list(dict.values(dictionary))   #convert a dict value to list
    dict_keys_list = list(dict.keys(dictionary))
    num_cluster = list(set(dict_value_list))

    num_clustering_list = []
    for x in range(len(num_cluster)):
        innerlist = []
        for i in range(len(dict_keys_list)):
            if dict_value_list[i] == num_cluster[x]:
                innerlist.append(dict_keys_list[i])
        num_clustering_list.append(innerlist)

    return num_clustering_list


def list_set_to_dict(list_set):
    
    partition_expand = sum(list_set, [])

    num_cluster = []
    for cluster in range(len(list_set)):
        for number in range(len(list_set[cluster])):
            num_cluster.append(cluster)

    dict = dict(zip(partition_expand, num_cluster))

    return dict



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
        mid = n_samples/2
        u_init[ind[:mid]] = 1
        u_init[ind[mid:]] = -1
        res =  u_init
    elif opt == 'rd':
        u_init = np.random.uniform(low = -1, high = 1, size = n_samples)
        res = u_init
    elif opt == 'eig':
        res =  V[:,1].copy()
    return res


def threshold(u, thre_val = 0):
    w = u.copy()
    w[w<thre_val] = 0
    w[w>thre_val] = 1
    return w
