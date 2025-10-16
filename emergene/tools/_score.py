import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy import sparse

from ._utils import _get_p_from_empi_null

#### Gene Set Scoring Method
def score(
    adata,
    gene_list,
    gene_weights=None,
    n_nearest_neighbors: int=30,
    leaf_size: int=40,
    layer: str='infog',
    random_seed: int=1927,
    n_ctrl_set:int=100,
    key_added:str=None,
    verbosity: int=0
):
    """
    For a given gene set, compute gene expression enrichment scores and P values for all the cells.

    Parameters
    ----------
    adata : AnnData
        The AnnData object for the gene expression matrix.
    
    gene_list : list of str
        A list of gene names for which the score will be computed.
    
    gene_weights : list of floats, optional
        A list of weights corresponding to the genes in `gene_list`. The length of 
        `gene_weights` must match the length of `gene_list`. If None, all genes in 
        `gene_list` are weighted equally. Default is None.
    
    n_nearest_neighbors : int, optional
        Number of nearest neighbors to consider for randomly selecting control gene sets based on the similarity of genes' mean and variance among all cells.
        Default is 30.
    
    leaf_size : int, optional
        Leaf size for the KD-tree or Ball-tree used in nearest neighbor calculations. Default is 40.
    
    layer : str, optional
        The name of the layer in `adata.layers` to use for gene expression values. Default is 'infog'.
    
    random_seed : int, optional
        Random seed for reproducibility. Default is 1927.
    
    n_ctrl_set : int, optional
        Number of control gene sets to be used for calculating P values. Default is 100.
    
    key_added : str, optional
        If provided, the computed scores will be stored in `adata.obs[key_added]`. The scores and P values will be stored in `adata.uns[key_added]` as well.
        Default is None, and the `INFOG_score` will be used as the key.
    verbosity : int, optional (default: 0)
        Level of verbosity for logging information.
        
    Returns
    -------
    None
        Modifies the `adata` object in-place, see `key_added`.
    """
    ### Set the random seed
    # Set the random seed for reproducibility
    np.random.seed(random_seed) 
    
    
    if gene_weights is None:
        gene_weights=np.repeat(1.0, len(gene_list))
    
    ### Calculate the mean and variance

    ### Calculate the variance
    # Determine the input matrix
    if layer is not None:
        cellxgene = adata.layers[layer] ### I think .copy() is not needed here
        ### For the query gene set
        cellxgene_subset=adata[:, gene_list].layers[layer]
    else:
        cellxgene = adata.X
        ### For the query gene set
        cellxgene_subset= adata[:, gene_list].X
    
    
    ### Calculate the mean and variance
    # c=cellxgene.copy()
    mean=np.array(cellxgene.mean(axis=0))
    infog_mean=mean.copy()[0]
    mean **=2
    ### Instead of c.data **= 2, I used cellxgene.multiply(cellxgene), which is the same result
    # c.data **= 2
    # residual_var_orig_b=np.squeeze(np.array(c.mean(axis=0))-mean) 
    residual_var_orig_b = np.squeeze(np.array(cellxgene.multiply(cellxgene).mean(axis=0)) - mean)
    
    mean_var=np.array([infog_mean, residual_var_orig_b]).T
    ### Construct a kNN graph for the genes based on gene means and gene variances 
    kdt = KDTree(mean_var, leaf_size=leaf_size, metric='euclidean')
    mean_var_knn_idx=kdt.query(mean_var, k=n_nearest_neighbors+1, return_distance=False)
    ### Remove the self node
    mask=mean_var_knn_idx != np.arange(mean_var.shape[0])[:,None]
    ### Use the mask to remove the self node
    mean_var_knn_idx=np.vstack(np.array([mean_var_knn_idx[i, mask[i]][: n_nearest_neighbors] for i in range(mean_var_knn_idx.shape[0])], dtype=np.int64))

    ### Only select for the query gene set
    mean_var_knn_idx_df=pd.DataFrame(mean_var_knn_idx)
    mean_var_knn_idx_df.index=adata.var_names.values
    gene_list_knn_idx=mean_var_knn_idx_df.loc[gene_list].values

    ### Create a matrix to hold the gene weights for randomly pickedup control genes

    # # Randomly select indices from each row
    # # This generates a matrix of shape (T, N) where each column contains random indices for the corresponding row
    # random_indices = np.random.randint(mean_var_knn_idx.shape[1], size=(n_ctrl_set, mean_var_knn_idx.shape[0]))
    n_genes=gene_list_knn_idx.shape[0]
    # Initialize an array to hold the sampled values
    n_ctrl_set_idx = np.empty((n_ctrl_set, n_genes), dtype=gene_list_knn_idx.dtype)
    ### Sampling genes with similar mean and variance 
    for n in range(n_genes):
        n_ctrl_set_idx[:,n] = np.random.choice(gene_list_knn_idx[n], size=n_ctrl_set, replace=True)


    ### Create a sparse matrix, rows are genes, columns are control gene set
    rows = []
    cols = []
    data = []
    for ctrl_i, ctrl_gene_idx in enumerate(n_ctrl_set_idx):
        rows.append(ctrl_gene_idx)
        cols.append(np.repeat(ctrl_i,len(gene_list)))
        data.append(gene_weights)

    ctrl_gene_weight = sparse.csr_matrix((np.ravel(data), (np.ravel(rows), np.ravel(cols))), shape=(adata.n_vars, n_ctrl_set))

    #### Apply L1-normalization as we need to calculate the mean value
    #### But it's not equal to L1-normalization, because the weight has it's own scale
    # ctrl_gene_weight=normalize(ctrl_gene_weight,norm='l1', axis=0)

    cellxgene_ctrl=cellxgene @ ctrl_gene_weight
    
    
    ### Need to do element-wise multiplication to add the gene weights:
    ### The following one is not correct, because the gene orders will be changed:
    ### cellxgene_query=np.ravel(cellxgene[:,np.isin(adata.var_names,gene_list)].multiply(gene_weights).mean(axis=1))
    # cellxgene_query=np.ravel(adata[:, gene_list].layers[layer].multiply(gene_weights).mean(axis=1))
    ## Use sum here, because the ctrl multiplication equals to sum
    ### cellxgene_subset is the cellxgene matrice with the input gene kept
    cellxgene_query=np.ravel(cellxgene_subset.multiply(np.array(gene_weights)).sum(axis=1))
    
    # Get p-values
    from statsmodels.stats.multitest import multipletests
    ### Should use >=, because a[i-1] < v <= a[i] is for left in numpy.searchsorted
    ### Refer to https://numpy.org/doc/2.1/reference/generated/numpy.searchsorted.html
    n_greater=np.sum(cellxgene_ctrl>= cellxgene_query[:, None], axis=1)
    p_value_monte_carlo = np.ravel( (n_greater+1) / (n_ctrl_set+1))
    nlog10_p_value_monte_carlo = -np.log10(p_value_monte_carlo)
    pooled_p_monte_carlo_FDR=multipletests(p_value_monte_carlo, method="fdr_bh")[1]
    nlog10_pooled_p_monte_carlo_FDR=-np.log10(pooled_p_monte_carlo_FDR)
    
    
    ### Caculate pool_valu
    pooled_p = _get_p_from_empi_null(cellxgene_query, cellxgene_ctrl.toarray().flatten())
    nlog10_pooled_p = -np.log10(pooled_p)
    
    pooled_p_FDR=multipletests(pooled_p, method="fdr_bh")[1]
    nlog10_pooled_p_FDR=-np.log10(pooled_p_FDR)
    

    BG=np.ravel(cellxgene_ctrl.mean(axis=1))
    
    ### Normalize the score by the number of genes and the gene weights
    scaling_factor=np.median(gene_weights)*len(gene_list)
    cellxgene_query=cellxgene_query/scaling_factor
    BG=BG/scaling_factor
    
    score= cellxgene_query - BG
    ### Use division
    # score= cellxgene_query/(BG + 1e-10) ## adding epsilon to avoid division by zero
    
    score_pval_res = {
                "score": score,
                "score_query": cellxgene_query,
                "score_ctrl_average": BG,
                "pval_mc": p_value_monte_carlo,
                "nlog10_pval_mc": nlog10_p_value_monte_carlo,
                "pval_mc_FDR": pooled_p_monte_carlo_FDR,
                "nlog10_pval_mc_FDR": nlog10_pooled_p_monte_carlo_FDR,
        
                "pval": pooled_p,
                "nlog10_pval": nlog10_pooled_p,
                "pval_FDR": pooled_p_FDR,
                "nlog10_pval_FDR": nlog10_pooled_p_FDR
            }

    df_score_pval_res = pd.DataFrame(index=adata.obs.index, data=score_pval_res, dtype=np.float32)
    
    if key_added is None:
        adata.obs['INFOG_score']=score
        adata.uns['INFOG_score']=df_score_pval_res
        if verbosity>0:
            print(f"Finished. The scores are saved in adata.obs['INFOG_score'] and the scores, P values are saved in adata.uns['INFOG_score'].")
        
    else:
        adata.obs[key_added]=score
        adata.uns[key_added]=df_score_pval_res
        if verbosity>0:
            print(f"Finished. The scores are saved in adata.obs['{key_added}'] and the scores, P values are saved in adata.uns['{key_added}'].")
          
            
