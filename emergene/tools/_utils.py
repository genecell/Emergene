from scipy.sparse import csr_matrix, issparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from anndata import AnnData
import scanpy as sc
from typing import Union, List


from sklearn.neighbors import kneighbors_graph
### Compute adjacency martix given the low-dimensional representations
def _build_adjacency_matrix(
    adata: AnnData,
    use_rep:str='X_pca',
    n_nearest_neighbors:int=10,
    sigma:float=100.0
    ):
    """
    Compute an adjacency matrix from low-dimensional cell representations.
    
    This function builds a k-nearest neighbors graph using the low-dimensional 
    representations stored in `adata.obsm[use_rep]`. It then converts the computed 
    distances into affinities using an exponential decay function.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object for preprocessed data.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` that corresponds to the low-dimensional representation to use.
    n_nearest_neighbors : int, optional (default: 10)
        Number of nearest neighbors to consider when constructing the graph.
    sigma : float, optional (default: 100.0)
        Scaling parameter for the exponential kernel. A larger sigma results in a slower 
        decay of weights.
    
    Returns
    -------
    cellxcell : scipy.sparse.csr_matrix
        Sparse adjacency matrix where the nonzero entries represent the affinity 
        (i.e., the transformed distance) between cells.
    """
    # Copy the low-dimensional representation to avoid modifying the original data.
    X_rep = adata.obsm[use_rep].copy()

    # Compute the k-nearest neighbors graph.
    cellxcell = kneighbors_graph(
        X_rep,
        n_neighbors=n_nearest_neighbors,
        mode='distance',
        include_self=False)
    # Convert the distance values to affinities using the exponential decay function.
    cellxcell.data=1/np.exp(cellxcell.data/sigma)
    
    return cellxcell


from typing import Union
def _calculate_rowwise_cosine_similarity(
    A: Union[np.ndarray, csr_matrix],
    B: Union[np.ndarray, csr_matrix]
):
    """
    Computes the cosine similarity between corresponding rows of matrices A and B.
    
    Cosine similarity between two vectors is defined as:
    
        cosine_similarity = (A[i] · B[i]) / (||A[i]|| * ||B[i]||)
    
    where "·" denotes the dot product and ||·|| is the Euclidean norm.
    
    This function computes the cosine similarity for each pair of corresponding rows 
    from matrices A and B. It supports both dense NumPy arrays and sparse CSR matrices.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse.csr_matrix, shape (n, m)
        First input matrix. Can be a dense array or a sparse CSR matrix.
    B : np.ndarray or scipy.sparse.csr_matrix, shape (n, m)
        Second input matrix, which must have the same shape as A.


    Returns
    -------
    np.ndarray, shape (n,)
        A 1D array where each element is the cosine similarity between the corresponding
        rows of A and B. If a row in either matrix has zero norm, the similarity for that
        row is defined to be zero.
    """

    if A.shape != B.shape:
        raise ValueError(f"Matrices A and B must have the same shape. Got A.shape = {A.shape}, B.shape = {B.shape}.")

    if issparse(A) and issparse(B):
        dot_products = A.multiply(B).sum(axis=1).A1  # A1 converts matrix to a flat array
        norm_A = np.sqrt(A.multiply(A).sum(axis=1)).A1
        norm_B = np.sqrt(B.multiply(B).sum(axis=1)).A1
    else:
        dot_products = np.einsum('ij,ij->i', A, B)  # It's faster than (A * B).sum(axis=1)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
    
    denominator = norm_A * norm_B
    cosine_similarities = np.divide(dot_products, denominator, out=np.zeros_like(dot_products), where=denominator!=0)
    
    return cosine_similarities


### To efficiently create a random adjacency matrix
def _build_random_adjacency_matrix(
    adata: AnnData,
    cellxcell: csr_matrix,
    n_nearest_neighbors:int=10,
    random_seed:int=0,
            ):
    """
    Create a random adjacency matrix by shuffling the connectivity weights of an existing cell-to-cell graph.

    This function takes an existing cell-to-cell adjacency matrix and creates a randomized version by shuffling the connectivity weights.

    Parameters
    ----------
    adata : AnnData
        AnnData object for preprocessed data.
    cellxcell : csr_matrix
        A cell-to-cell adjacency matrix where the `.data` attribute contains the connectivity weights.
    n_nearest_neighbors : int, optional (default: 10)
        Number of nearest neighbors per cell in the original adjacency matrix.
    random_seed : int, optional (default: 0)
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    csr_matrix
        A new cell-to-cell adjacency matrix with shuffled connectivity weights, maintaining the same shape and number of nonzero entries as the original.
    """
    
    row_ind=np.repeat(np.arange(adata.n_obs), n_nearest_neighbors)
    np.random.seed(random_seed)
    col_ind=np.random.choice(np.arange(adata.n_obs), adata.n_obs*n_nearest_neighbors, replace=True)
    
    ### Shuffle the weights
    connectivity=cellxcell.data.copy()
    np.random.shuffle(connectivity)

    cellxcell_shuffle=csr_matrix((connectivity,(row_ind, col_ind)),shape=cellxcell.shape)
    
    return cellxcell_shuffle


from sklearn.preprocessing import normalize
def _generate_random_background(
    adata: AnnData,
    cellxcell: csr_matrix,
    genexcell: csr_matrix,
    n_nearest_neighbors:int=10,
    n_repeats:int=30,
    random_seed:int=0,
    
):
    """
    Generate a background distribution based on shuffled gene expression patterns.

    Parameters
    ----------
    adata : AnnData
        AnnData object for preprocessed data.
    cellxcell : csr_matrix
        The original cell-to-cell adjacency matrix. The nonzero entries in its `.data` attribute represent the connectivity weights.
    genexcell : csr_matrix
        Gene expression matrix in sparse format with shape (n_genes, n_cells). Each row corresponds to a gene and each column to a cell.
    n_nearest_neighbors : int, optional
        Number of nearest neighbors per cell used when generating the randomized adjacency matrices.
    n_repeats : int, optional
        Number of randomizations to perform.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    List[np.ndarray]
        A list where each element is a 1D numpy array.
    """
    
    random_gsp_list=[]
    
    np.random.seed(random_seed)
    # Generate a list of unique random seeds for each repeat.
    random_seed_list=np.random.choice(1902772, size=n_repeats, replace=False)
    
    for i in np.arange(n_repeats):
        random_seed=random_seed_list[i]
    
        cellxcell_shuffle=_build_random_adjacency_matrix(
            adata,
            cellxcell,
            n_nearest_neighbors=n_nearest_neighbors,
            random_seed=random_seed)
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_shuffle=normalize(cellxcell_shuffle, axis=1, norm='l1')
        
    
        random_order1=genexcell @ cellxcell_shuffle.T


        random_gsp_list.append(_calculate_rowwise_cosine_similarity(genexcell, random_order1))
    
    return(random_gsp_list)


def _build_adjacency_matrix_acrossDataset(
    adata: AnnData,
    use_rep:str='X_pca',
    condition_key:str='Sample',
    n_nearest_neighbors:int=10,
            ):
    """
    Build an adjacency (connectivity) matrix across datasets using the bbknn algorithm.

    Parameters
    ----------
    adata : AnnData
        AnnData object for preprocessed data.
    use_rep : str, optional (default: 'X_pca')
        Key in `adata.obsm` that holds the low-dimensional representation to be used.
    condition_key : str, optional (default: 'Sample')
        Key in `adata.obs` that indicates the batch or condition label for each cell.
    n_nearest_neighbors : int, optional (default: 10)
        Number of nearest neighbors to consider within each batch when running bbknn.

    Returns
    -------
    csr_matrix
        Sparse cell-to-cell connectivity (adjacency) matrix computed across datasets.
    """
    
    # X_rep=adata.obsm[use_rep].copy()
    
    adata_copy=sc.external.pp.bbknn(
        adata,
        batch_key=condition_key,
        use_rep=use_rep,
        copy=True,
        neighbors_within_batch=n_nearest_neighbors
    )
    cellxcell_acrossDataset=adata_copy.obsp['connectivities']
    
    return cellxcell_acrossDataset



### Refer to Scanpy for _select_top_n function
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices



### Refer to scDRS for _get_p_from_empi_null function
def _get_p_from_empi_null(v_t, v_t_null):
    """Compute p-value from empirical null
    For score T and a set of null score T_1,...T_N, the p-value is

        p= [1 + \Sigma_{i=1}^N 1_{ (T_i \geq T) }] / (1+N)

    If T, T_1, ..., T_N are i.i.d. variables following a null distritbuion,
    then p is super-uniform.

    The naive algorithm is N^2. Here we provide an O(N log N) algorithm to
    compute the p-value for each of the N elements in v_t

    Args
    ----
    v_t : np.ndarray
        Observed score of shape (M,).
    v_t_null : np.ndarray
        Null scores of shape (N,).
        
    Returns
    -------
    v_p: : np.ndarray
        P-value for each element in v_t of shape (M,).
    """

    v_t = np.array(v_t)
    v_t_null = np.array(v_t_null)

    v_t_null = np.sort(v_t_null)
    v_pos = np.searchsorted(v_t_null, v_t, side="left")
    v_p = (v_t_null.shape[0] - v_pos + 1) / (v_t_null.shape[0] + 1)
    return v_p
