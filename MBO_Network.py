import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigs, eigsh

from graph_mbo.utils import apply_threshold, get_fidelity_term, get_initial_state




"""
    Run the MBO scheme on a graph.
    Parameters
    ----------
    adj_matrix : np.array
        The adjacency matrix of the graph.
    normalized : bool
        Use the normalized graph Laplacian.
    signless : bool
        Use the signless graph Laplacian to find eigenvalues if normalized
    pseudospectral : bool
        Use the pseudospectral solver. If false, use CG or LU.
    m : int
        Number of eigenvalues to use for pseudospectral
    num_communities : int
        Number of communities
    target_size : list
        List of desired community sizes when using auction MBO
    thresh_type : str
        Type of thresholding to use. "max" takes the max across communities,
        "auction" does auction MBO
    dt : float
        Time step between thresholds for the MBO scheme
    min_dt : float
        Minimum time step for MBO convergence
    max_iter : int
        Maximum number of iterations
    n_inner : int
        Number of iterations for the MBO diffusion loop
    """

def mbo_modularity_eig(num_communities, m, dt, adj_matrix, tol ,inner_step_count, normalized=True,symmetric=True,
                       pseudospectral=True, target_size=None, signless=None, fidelity_type="karate", max_iter=10000,fidelity_coeff=10,
                       initial_state_type="fidelity", thresh_type="max"): # inner stepcount is actually important! and can't be set to 1...
    
    A_absolute_matrix = np.abs(adj_matrix)
    degree = np.array(np.sum(A_absolute_matrix, axis=-1)).flatten()
    num_nodes = len(degree)
    
    if num_nodes < num_communities:
        print("Wrong input, N should larger or equal than K.")
    
    else:
        
        m = min(num_nodes - 2, m)  # Number of eigenvalues to use for pseudospectral

        if target_size is None:
            target_size = [num_nodes // num_communities for i in range(num_communities)]
            target_size[-1] = num_nodes - sum(target_size[:-1])

        #graph_laplacian, degree = sp.sparse.csgraph.laplacian(A_absolute_matrix, return_diag=True)
        degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
        graph_laplacian = degree_diag - adj_matrix
        if symmetric:
            degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
            graph_laplacian = np.sqrt(degree_inv) @ graph_laplacian @ np.sqrt(degree_inv)
            # degree = np.ones(num_nodes)
            # degree_diag = sp.sparse.spdiags([degree], [0], num_nodes, num_nodes)
        elif normalized:
            degree_inv = sp.sparse.spdiags([1.0 / degree], [0], num_nodes, num_nodes)
            graph_laplacian = degree_inv @ graph_laplacian
        

        if pseudospectral:
            #if signless:
            #    if normalized:
            #        pass
            #    else:
            #        graph_laplacian = 2 * degree_diag - graph_laplacian
            if normalized:
                D, V = eigs(
                    graph_laplacian,
                    k=m,
                    v0=np.ones((graph_laplacian.shape[0], 1)),
                    which="LR" if signless else "SR",
                )
            else:
                D, V = eigsh(
                    graph_laplacian,
                    k=m,
                    v0=np.ones((graph_laplacian.shape[0], 1)),
                    which="LA" if signless else "SA",
                )
            #if signless:
            #    D = 2 * np.ones((m,)) - D  # Change D to be eigenvalues of graph Laplacian
            if normalized:
                # rescale eigenvectors to normalized space and orthogonalize
                for i in range(len(D)):
                    V[:, i] /= np.sqrt(V[:, i].transpose() @ degree_diag @ V[:, i])
        
        last_dt = 0

        if fidelity_type == "spectral":
            fidelity_D, fidelity_V = eigsh(
                graph_laplacian,
                k=num_communities + 1,
                v0=np.ones((graph_laplacian.shape[0], 1)),
                which="SA",
            )
            fidelity_V = fidelity_V[:, 1:]  # Remove the constant eigenvector
            fidelity_D = fidelity_D[1:]
            # apply_threshold(fidelity_V, target_size, "max")
            # return fidelity_V
        else:
            fidelity_V = None


        # Initialize parameters
        u = get_initial_state(
            num_nodes,
            num_communities,
            target_size,
            type=initial_state_type,
            fidelity_type=fidelity_type,
            fidelity_V=fidelity_V,)


        last_last_index = u == 1
        last_index = u == 1
        last_dt = 0
        
        
        # Perform MBO scheme

        for n in range(max_iter):
            dti = dt / inner_step_count

            if pseudospectral:
                if normalized:
                    a = V.transpose() @ (degree_inv @ u)  # Project into Hilbert space
                else:
                    a = V.transpose() @ u
                d = np.zeros((m, num_communities))
                demon = sp.sparse.spdiags([1 / (1 + dti * D)], [0], m, m)
            else:
                if last_dt != dt:
                    lu, piv = lu_factor(sp.sparse.eye(num_nodes) + dti * graph_laplacian)

            
            for j in range(inner_step_count):
                # Solve system (apply CG or pseudospectral)
                if pseudospectral:
                    a = demon @ (a + fidelity_coeff * dti * d)
                    u = V @ a  # Project back into normal space
                    fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                    # Project fidelity term into Hilbert space
                    if normalized:
                        d = V.transpose() @ (degree_inv @ fidelity_term)
                    else:
                        d = V.transpose() @ fidelity_term
                else:
                    fidelity_term = get_fidelity_term(u, type=fidelity_type, V=fidelity_V)
                    u += fidelity_coeff * dti * fidelity_term
            
                    for i in range(num_communities):
                        u[:, i] = lu_solve((lu, piv), u[:, i])

                j = j + 1
                

            # Apply thresholding 
            apply_threshold(u, target_size, thresh_type)

            # Stopping criterion 
            # Check that the index is changing and stop if time step becomes too small
            index = u == 1
            last_dt = dt

            norm_deviation = sp.linalg.norm(last_index ^ index) / sp.linalg.norm(index)
            if norm_deviation < tol :
                if dt < tol:
                    break
                else:
                    dt *= 0.5
            elif np.sum(last_last_index ^ index) == 0:
                # Going back and forth
                dt *= 0.5
            last_last_index = last_index
            last_index = index
            
            n = n+1

        if dt >= tol:
            print("MBO failed to converge")
        return u


# generate a random graph with community structure by the signed stochastic block model 
def SSBM_own(N, K):
    if N%K != 0:
        print("Wrong Input")

    else:
        s_matrix = -np.ones((N,N))
        cluster_size = N/K
        clusterlist = []
        for cs in range(K):
            clusterlist.append(int(cs*cluster_size))
        clusterlist.append(int(N))
        clusterlist.sort()
        #print(clusterlist)

        accmulate_size = []
        for quantity in range(len(clusterlist)-1):
            accmulate_size.append(clusterlist[quantity+1]-clusterlist[quantity])
        #print(accmulate_size)

        for interval in range(len(clusterlist)):
            for i in range(clusterlist[interval-1], clusterlist[interval]):
                for j in range(clusterlist[interval-1], clusterlist[interval]):
                    s_matrix[i][j] = 1
        #print(s_matrix)

        ground_truth = []
        for gt in range(len(accmulate_size)):
            ground_truth.extend([gt for y in range(accmulate_size[gt])])
        #print(ground_truth)

        ground_truth_v2 = []
        for gt in range(len(accmulate_size)):
            ground_truth_v2.extend([accmulate_size[gt] for y in range(accmulate_size[gt])])
        #print(ground_truth_v2)

    return s_matrix, ground_truth



def data_generator(s_matrix, noise, sparsity):
    A_init_matrix = s_matrix
    N = s_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            if i >= j:
                A_init_matrix[i][j] = 0
            if i < j:
                elements = [A_init_matrix[i][j], 0, -A_init_matrix[i][j]]
                probabilities = [(1- noise)*sparsity, 1-sparsity, noise*sparsity]
                A_init_matrix[i][j] = np.random.choice(elements, 1, p=probabilities)
    A_matrix = A_init_matrix + A_init_matrix.T - np.diag(np.diag(A_init_matrix))
    return A_matrix
