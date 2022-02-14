import networkx as nx
import numpy as np
import scipy as sp
import random
from networkx.algorithms.community import modularity
from scipy.sparse.linalg import eigsh
from itertools import product


def get_initial_state(
    num_nodes,
    num_communities,
    target_size,
    type="random",
    fidelity_type=None,
    fidelity_V=None,
):
    u = np.zeros((num_nodes, num_communities))
    if type == "random":
        for i in range(num_communities - 1):
            count = 0
            while count < target_size[i]:
                rand_index = np.random.randint(0, num_nodes - 1)
                if u[rand_index, i] == 0:
                    u[rand_index, i] = 1
                    count += 1
        u[np.sum(u, axis=1) < 1, -1] = 1
    elif type == "fidelity":
        return get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
    elif type == "fidelity_avg":
        u[:] = 1.0 / num_communities
        fidelity = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
        u[np.sum(fidelity, axis=1) > 0] = fidelity[np.sum(fidelity, axis=1) > 0]
    elif type == "spectral":
        u[:] = fidelity_V
        apply_threshold(u, None, "max")
    return u


def get_initial_state_1(num_nodes,num_communities):
    u_init = np.zeros((num_nodes, num_communities))   
    # if type is binary (e.g. karate club)
    u = np.zeros(u_init.shape)
    u[0, 0] = 1 - u_init[0, 0]
    u[0, 1] = -u_init[0, 1]
    u[-1, -1] = 1 - u_init[-1, -1]
    u[-1, 0] = -u_init[-1, 0]

    #for i in range(num_communities - 1):
    #    count = 0
    #    while count < target_size[i]:
    #        rand_index = np.random.randint(0, num_nodes - 1)
    #        if u[rand_index, i] == 0:
    #            u[rand_index, i] = 1
    #            count += 1
    #u[np.sum(u, axis=1) < 1, -1] = 1

    # Ensure each cluster has at least one node
    # Generate data list that store one node for each cluster
    #K_force = random.sample(range(num_communities),num_communities)
    
    # Random select rows with the amount of clusters
    #K_row = random.sample(range(num_nodes),num_communities)
    #for j in range(num_communities):

        # Force the selected row to be zero vector
    #    u[K_row[j],:] = np.zeros(num_communities)

        # Store the data list determined node to this cluster
    #    u[K_row[j],K_force[j]] = 1

    return u


def get_fidelity_term(u, type="karate", V=None):
    fidelity_term = np.zeros(u.shape)
    if type == "karate":
        fidelity_term[0, 0] = 1 - u[0, 0]
        fidelity_term[0, 1] = -u[0, 1]
        fidelity_term[-1, -1] = 1 - u[-1, -1]
        fidelity_term[-1, 0] = -u[-1, 0]
    elif type == "spectral":
        # Use the top component of each eigenvector to seed the Clusters
        if V is None:
            raise Exception()
        idxs = np.argmax(V, axis=0)
        # fidelity_term[idxs, :] = -1.0 / (u.shape[1]-1)
        fidelity_term[idxs, range(u.shape[1])] = 1  # - u[idxs, range(u.shape[1])]
    return fidelity_term


def apply_threshold(u, target_size, thresh_type):
    if thresh_type == "max":
        """Threshold to the max value across communities. Ignores target_size"""
        max_idx = np.argmax(u, axis=1)
        u[:, :] = np.zeros_like(u)
        u[(range(u.shape[0]), max_idx)] = 1
    elif thresh_type == "auction":
        """Auction between classes until target sizes are reached"""
        prices = np.zeros((1, u.shape[1]))  # Current price of community
        assignments = np.zeros_like(u)  # 1 where assigned, 0 elsewhere
        bids = np.zeros((u.shape[0],))  # Bid of each node
        epsilon = 0.01
        while np.sum(assignments) != np.sum(
            target_size
        ):  # Check if all targets are satisfied
            unassigned = np.argwhere(np.sum(assignments, axis=1) < 1)[:, 0]
            for x in unassigned:
                profit = u[x, :] - prices
                # ics = np.argmax(u[x, :]-prices)
                ics = np.flatnonzero(profit == profit.max())
                i = np.random.choice(ics)
                profit = np.delete(profit, i)
                i_next = np.random.choice(np.flatnonzero(profit == profit.max()))
                if i_next >= i:
                    i_next += 1
                price_diff = u[x, i] - prices[0, i]
                price_diff_next = u[x, i_next] - prices[0, i_next]
                bids[x] = prices[0, i] + epsilon + price_diff - price_diff_next
                if np.sum(assignments[:, i]) == target_size[i]:
                    assigned = np.argwhere(assignments[:, i] > 0)[:, 0]
                    y = np.argmin(bids[assigned])
                    y = assigned[y]
                    assignments[y, i] = 0
                    assignments[x, i] = 1
                    prices[0, i] = np.min(bids[assignments[:, i] > 0])
                else:
                    assignments[x, i] = 1
                    if np.sum(assignments[:, i]) == target_size[i]:
                        prices[0, i] = np.min(bids[assignments[:, i] > 0])
        # If there are any remaining, do max assignment
        max_idx = np.argmax(u, axis=1)
        unassigned = np.argwhere(np.sum(assignments, axis=1) == 0).flatten()
        assignments[(unassigned, max_idx[unassigned])] = 1
        u[:, :] = assignments
    return u


def get_modularity_original(adj, u):
    """Calculate the modularity score for the given community structure"""
    nxgraph = nx.convert_matrix.from_numpy_matrix(adj, create_using=nx.Graph())
    communities = [np.argwhere(u[:, i]).flatten() for i in range(u.shape[1])]
    return modularity(nxgraph, communities)


def spectral_clustering(adj, num_communities):   # adjacancy matrix > 0
    degree = np.array(np.sum(adj, axis=1)).flatten()
    num_nodes = len(degree)
    graph_laplacian, degree = sp.sparse.csgraph.laplacian(adj, return_diag=True)
    #degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
    degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
    sym_graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)
    D, V = eigsh(
        sym_graph_laplacian,
        k=num_communities + 1,
        v0=np.ones((sym_graph_laplacian.shape[0], 1)),
        which="SA",
    )
    V = V[:, 1:]
    apply_threshold(V, None, "max")
    return V



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


def vector_to_labels(V):
    """  convert a multiclass assignment vector (n_samples, n_class) 
    to a standard label 0,1,2... (n_samples,) by projecting onto largest component

    Parameters
    -----------
    V : ndarray, shape(n_samples,n_class)
        class assignment vector for multiclass

    """
    return np.argmax(V, axis = 1)


def get_modularity(network, community_dict):
    '''
    Calculate the modularity. Edge weights are ignored.

    Undirected:
    .. math:: Q = \frac{1}{2m}\sum_{i,j} \(A_ij - \frac{k_i k_j}{2m}\) * \detal_(c_i, c_j)

    Directed:
    .. math:: Q = \frac{1}{m}\sum_{i,j} \(A_ij - \frac{k_i^{in} k_j^{out}}{m}\) * \detal_{c_i, c_j}

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    community_dict : dict
        A dictionary to store the membership of each node
        Key is node and value is community index

    Returns
    -------
    float
        The modularity of `network` given `community_dict`
    '''

    Q = 0
    G = network.copy()
    nx.set_edge_attributes(G, {e:1 for e in G.edges}, 'weight')
    A = nx.to_scipy_sparse_matrix(G).astype(float)

    if type(G) == nx.Graph:
        # for undirected graphs, in and out treated as the same thing
        out_degree = in_degree = dict(nx.degree(G))
        M = 2.*(G.number_of_edges())
        #print("Calculating modularity: ")
    #elif type(G) == nx.DiGraph:
    #    in_degree = dict(G.in_degree())
    #    out_degree = dict(G.out_degree())
    #    M = 1.*G.number_of_edges()
    #    print("Calculating modularity for directed graph")
    else:
        print('Invalid graph type')
        raise TypeError

    nodes = list(G)
    Q = np.sum([A[i,j] - in_degree[nodes[i]]*\
                         out_degree[nodes[j]]/M\
                 for i, j in product(range(len(nodes)),\
                                     range(len(nodes))) \
                if community_dict[nodes[i]] == community_dict[nodes[j]]])
    return Q / M



def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()


def _mbo_forward_step_multiclass(u): #thresholding
    return labels_to_vector(vector_to_labels(u),vec_dim = u.shape[1])

def label_to_dict(u_label):
    len_label = []
    for i in range(len(u_label)):
        len_label.append(i)
    u_dict = dict(zip(len_label, u_label))
    return u_dict