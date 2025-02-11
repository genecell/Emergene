import numpy as np
from scipy.sparse import csr_matrix, issparse
import scanpy as sc
from anndata import AnnData



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


from scipy import sparse
from sklearn.preprocessing import normalize
from typing import List, Optional

def EmerGene(
    adata: AnnData,
    use_rep:str='X_pca',
    use_rep_acrossDataset:str='X_pca',
    layer: Optional[str] = None,
    n_nearest_neighbors:int=10,
    condition_key:str='Sample',
    random_seed:int=27,
    n_repeats:int=3,

    mu: float = 1.0,
    beta: float = 1.0,
    sigma: float = 100.0,
    
    n_cells_expressed_threshold: int = 50,
    n_top_EG_genes: int = 500,
    
    remove_lowly_expressed=True,
    expressed_pct: float = 0.1,
    
    inplace: bool = False,
    gene_list_as_string: bool = False,
    verbose:int=1,

):
    """
    Compute EmerGene scores and local fold-change matrices for genes across different conditions.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object for preprocessed data.
    use_rep : str, optional
        Key in `adata.obsm` for the low-dimensional embedding used for condition-specific diffusion (default: 'X_pca').
    use_rep_acrossDataset : str, optional
        Key in `adata.obsm` for computing the across-dataset connectivity matrix (default: 'X_pca').
    layer : Optional[str], optional
        Key in `adata.layers` representing the gene expression matrix. If `None`, the function uses 
        the default expression matrix stored in `adata.X`. (default: None)
    n_nearest_neighbors : int, optional
        Number of nearest neighbors used when constructing adjacency matrices (default: 10).
    condition_key : str, optional
        Key in `adata.obs` that specifies the condition (or batch) label for each cell (default: 'Sample').
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility (default: 27).
    n_repeats : int, optional
        Number of randomizations to perform for background generation (default: 3).
    mu : float, optional
        Weight for subtracting the random background specificity in the final EmerGene score (default: 1.0).
    beta : float, optional
        Weight for subtracting the condition-wise background specificity in the final EmerGene score (default: 1.0).
    sigma : float, optional
        Parameter for scaling in the adjacency matrix construction (default: 100.0).
    n_cells_expressed_threshold : int, optional
        Threshold for the number of cells expressing a gene (default: 50).
    n_top_EG_genes : int, optional
        Number of top EmerGene genes to select for output (default: 500).
    remove_lowly_expressed : bool, optional
        Flag indicating whether to remove lowly expressed genes (currently not implemented) (default: True).
    expressed_pct : float, optional
        Minimum percentage of cells in which a gene must be expressed (currently not implemented) (default: 0.1).
    inplace : bool, optional
        If True, saves EmerGene scores into `adata.var`. If False, returns a pandas DataFrame with the scores (default: False).
    gene_list_as_string: bool, optional
        If True, save the genes and scores as a string. If False, save as a pandas DataFrame with two columns for genes and scores separately.
    verbose : int, optional
        Verbosity level; if > 0, progress messages will be printed (default: 1).
    
    Returns
    -------
    If `inplace` is False, returns a tuple containing:
      - A Dictionary of the DataFrames of top gene sets, with keys are the conditions.
      - A DataFrame where each column is named `EmerGene_{condition}` with the corresponding EmerGene 
        scores for all genes.
    If `inplace` is True, returns the top gene set Dictionary and modifies `adata` in-place.
    """
    
    
    ### Record the condition info
    condition_info=adata.obs[condition_key]
    conditions=np.unique(condition_info)
    n_conditions=len(conditions)
    
    # Extract gene expression data from the specified layer or default to adata.X if layer is None.
    if layer is None:
        genexcell = adata.X.T
    else:
        genexcell = adata.layers[layer].T
        
    
    # Initialize output structures
    EG_dict = {}  # Dictionary for condition-wise DataFrames
    
    # If not inplace, initialize a DataFrame to store EmerGene scores.
    if not inplace:
        emergene_scores = pd.DataFrame(index=adata.var_names)
        
    ### Calculate the number of expressed cells for each gene
    # Check if adata.X is sparse. 
    if issparse(adata.X):
        # For sparse matrices, use getnnz along axis=0 (columns represent genes).
        n_cells_expressed = adata.X.getnnz(axis=0)
    else:
        # For dense matrices, count the number of cells where expression is > 0.
        n_cells_expressed = np.sum(adata.X > 0, axis=0)

    
    # Create a copy of genexcell to modify
    # genexcell_foldchange = genexcell.A.copy()
    ### Create a list to store all the sparse matrix, and then reorder it later
    fold_change_list=list()
    index_list=list()
    
 
    cellxcell_acrossDataset=_build_adjacency_matrix_acrossDataset(adata, use_rep=use_rep_acrossDataset, condition_key=condition_key, n_nearest_neighbors=n_nearest_neighbors)
    
    
    for condition in conditions:
        if verbose>0:
            print('Processing condition: ', condition)
        ### get the info of the target condition
        idx_cells = np.where(condition_info == condition)[0]

        cellxcell_target=_build_adjacency_matrix(
            adata[condition_info == condition],
            use_rep=use_rep,
            n_nearest_neighbors=n_nearest_neighbors,
            sigma=sigma,
                      )
        
        
        ### Calculate the target condition specificity
        genexcell_target=genexcell[:,idx_cells]
        
        ### Calculate the target condition's background specificity
        ### Build the random background by sampling
        random_gsp_list=_generate_random_background(adata[condition_info == condition], cellxcell_target, genexcell_target,
                            n_nearest_neighbors=n_nearest_neighbors,
                            n_repeats=n_repeats,
                            random_seed=random_seed)
        random_gsp=np.mean(random_gsp_list,axis=0)
        
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_target=normalize(cellxcell_target, axis=1, norm='l1')
        ### Diffused on the same dataset
        genexcell_target_diffusion=genexcell_target @ cellxcell_target.T
       
        gsp=_calculate_rowwise_cosine_similarity(genexcell_target, genexcell_target_diffusion)

        


        ### Calculate the condition-wise specificty
        idx_remaining_cells = np.where(condition_info != condition)[0]
        cellxcell_targetXremaining=cellxcell_acrossDataset[idx_cells,:][:,idx_remaining_cells]
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_targetXremaining=normalize(cellxcell_targetXremaining, axis=1, norm='l1')

        genexcell_remaining=genexcell[:,idx_remaining_cells]
        
        genexcell_conditionwise_diffusion=genexcell_remaining @ cellxcell_targetXremaining.T
        
        gsp_conditionWise=_calculate_rowwise_cosine_similarity(genexcell_target, genexcell_conditionwise_diffusion)
        
        
        ### Record the fold change
        # Create a copy of B to avoid modifying the original matrix
        inv_b = genexcell_conditionwise_diffusion.copy()
        # Invert non-zero elements safely. Zeros in B remain zero in inv_b, which acts like ones in division. Then minus 1.0, this is used to combine with add genexcell_target itself, then for the values nonzero in A, but zero in B, keep it unchanged in A during the multiplication
        inv_b.data = 1.0 / inv_b.data -1.0
        # Reset the genexcell_foldchange, and will return it back after the calculation
        # genexcell_foldchange[:,idx_cells] = 
        ## use genexcell_target_diffusion instead of genexcell_target, to represent the local average expression
        fold_change_list.append(genexcell_target_diffusion.multiply(inv_b) + genexcell_target_diffusion)
        index_list.append(idx_cells)

        gsp_score=gsp-mu*random_gsp-beta*gsp_conditionWise
        gsp_score=np.array(gsp_score).ravel()
        
        
        ### Filter by number_of_expressed_cells
        ### Could improve this by number_of_expressed_cells in the adjacent neighbors, could do this easily by element-wise
        ### - multiplication 
        gsp_score[n_cells_expressed < n_cells_expressed_threshold] = gsp_score.min() - 1e-6

        ### Select top genes based on EmerGene scores
        EG_genes_idx=_select_top_n(gsp_score, n_top_EG_genes)
        EG_genes_list=adata.var_names.values[EG_genes_idx]
        EG_genes_list_scores=np.round(gsp_score[EG_genes_idx], 6)
        
        ### Store result in dictionary
        if gene_list_as_string:
            EG_dict[f'EG_{condition}'] = ",".join(f"{gene}:{score:.6f}" for gene, score in zip(EG_genes_list, EG_genes_list_scores))
        else:
            EG_dict[f'EG_{condition}'] = pd.DataFrame({'Gene': EG_genes_list, 'EG_score': EG_genes_list_scores})
        
        # Save the EmerGene scores.
        if inplace:
            adata.var[f'EmerGene_{condition}'] = gsp_score
            if verbose > 0:
                print(f"The EmerGene scores for all genes in condition {condition} are saved in adata.var['EmerGene_{condition}'] in `adata.var`.")
        else:
            emergene_scores[f'EmerGene_{condition}'] = gsp_score
            if verbose > 0:
                print(f"The EmerGene scores for all genes in condition {condition} are saved in the dataframe column 'EmerGene_{condition}'.")
                
        
    ### Save the result as a layer in adata    
    original_indice=np.hstack(index_list)
    ###original_indice
    new_indice=range(len(original_indice))
    # Sort the dictionary by key
    sorted_dict = dict(sorted(dict(zip(original_indice, new_indice)).items()))
    reorder_indice=list((sorted_dict.values()))
    
    merged_sparse_matrix=sparse.hstack(fold_change_list)
    merged_sparse_matrix=merged_sparse_matrix[:,reorder_indice].T
    merged_sparse_matrix.data=np.log1p(merged_sparse_matrix.data)
    adata.layers['localFC']=merged_sparse_matrix
    if verbose>0:
        print("The local foldchange information for each individual cells is saved as `localFC` in adata.layers.")
        print("Finished running the EmerGene.")

    # Return the EmerGene scores DataFrame if not modifying adata in-place.
    if not inplace:
        return EG_dict, emergene_scores
    else:
        return EG_dict
    
    
#### To calculate p-value and score
import scdrs
import scipy as sp
import pandas as pd
from scipy.sparse import csr_matrix
## L1 normalization
from sklearn.preprocessing import normalize

def computeScore(
    adata,
    geneset_dict,
    layer=None,
    n_ctrl:int=1000,
    ctrl_match_key='mean_var',
    n_genebin:int=200,
    n_mean_bin:int=20,
    n_var_bin:int=20,
    weight_opt:str='vs',
    return_ctrl_raw_score:bool=False,
    return_ctrl_norm_score:bool=False,
    random_seed:int=27,
    verbose:int=0,
):
    ### Could replace this with the idea of ChromVAR or pyChromVAR, to construct a gene graph, then the sampling of genes would also be easy
    ### I used the sampling of genes, when I do the GWAS enrich analysis, please double check that code as well
    if layer is not None:
        X_tmp=adata.X.copy()
        adata.X=adata.layers[layer]
    
    valid_genes = set(adata.var_names)
 
    scdrs.preprocess(adata, n_mean_bin=n_mean_bin, n_var_bin=n_var_bin, copy=False)

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[adata.var_names].copy()
    df_gene["gene"] = df_gene.index
    df_gene.drop_duplicates(subset="gene", inplace=True)
    
    
    
    
    ### To save the results for each geneset
    dict_res=dict()
    print("The list of groups to process: ", list(geneset_dict.keys()))
    for group, gene_and_weight in geneset_dict.items():
        print("Processing :", group)
        
        # Parse and filter the dictionary in one step, only keep the genes in adata.var_names
        gene_dict = {k: float(v) for k, v in (item.split(":") for item in gene_and_weight.split(",")) if k in valid_genes}

        gene_list=list(gene_dict.keys())
        gene_weight=list(gene_dict.values())


        dict_ctrl_list, dict_ctrl_weight = scdrs.method._select_ctrl_geneset(df_gene,
                                                                           gene_list,
                                                                           gene_weight,
                                                                           ctrl_match_key=ctrl_match_key,
                                                                           n_ctrl=n_ctrl,
                                                                           n_genebin=n_genebin,
                                                                           random_seed=random_seed)

        ### Creat a sparse matrix for control gene set's gene indice and gene weight
        gene2indice=dict(zip(adata.var_names.values, range(adata.n_vars)))

        ### Create a sparse matrix, rows are genes, columns are control gene set
        rows = []
        cols = []
        data = []
        for ctrl_i, ctrl_i_gene in dict_ctrl_list.items():
            ### local dict to save the correspondence between gene names and the gene weights
            local_gene_weight_dict=dict(zip(ctrl_i_gene,dict_ctrl_weight[ctrl_i]))
            for gene_ in ctrl_i_gene:
                rows.append(gene2indice[gene_])
                cols.append(ctrl_i)
                data.append(local_gene_weight_dict[gene_])
        contrl_gene_weight = csr_matrix((data, (rows, cols)), shape=(adata.n_vars, len(dict_ctrl_list)))

  

        ### Calculate the v_score_weight for all the genes
        if weight_opt == "vs":
            v_score_weight = 1 / np.sqrt(df_gene.loc[:, "var_tech"].values + 1e-2)

        df_gene['vs']=v_score_weight
        dict_v_score_weight=dict(zip(df_gene['gene'].values, v_score_weight))
        v_score_weight=np.ravel([dict_v_score_weight[i] for i in adata.var_names])



        #### To calculate the weight martix
        #### For sparse matrix
        contrl_gene_weight2=contrl_gene_weight.T.multiply(csr_matrix(v_score_weight)).T
        ### Normalization
        contrl_gene_weight2=normalize(contrl_gene_weight2,norm='l1',axis=0)


        ### The corresponding values for the input gene set, calculate a l1-normalized vector
        v_score_weight_input_geneset = np.multiply(df_gene.loc[gene_list, "vs"].values, gene_weight)
        v_score_weight_input_geneset = v_score_weight_input_geneset / v_score_weight_input_geneset.sum()
        v_raw_score_geneset=adata[:, gene_list].X @ v_score_weight_input_geneset

        ##############
        #### Variance adjustment
        ##############
        ### Create a sparse matrix filled with gene's variance on the control genes' positions
        contrl_gene_idx=contrl_gene_weight.copy()
        contrl_gene_idx.data=np.ones_like(contrl_gene_idx.data) ### Just to get the indice of the gene positions
        contrl_gene_var=contrl_gene_idx.T.multiply(csr_matrix(df_gene['var'].values)).T   

        ### Get the mat_ctrl_weight[:, i_ctrl] ** 2
        contrl_gene_weight2_var=contrl_gene_weight2.copy()
        contrl_gene_weight2_var.data=contrl_gene_weight2_var.data**2
        ### Element-wise multiplication
        contrl_gene_var=contrl_gene_var.multiply(contrl_gene_weight2_var)

        ### Normalize by the input gene list' values
        v_var_ratio_c2t=contrl_gene_var.sum(axis=0)/(df_gene.loc[gene_list, "var"] * v_score_weight_input_geneset ** 2).sum()
        v_var_ratio_c2t=np.ravel(v_var_ratio_c2t)


        ####
        mat_ctrl_raw_score=(adata.X @ contrl_gene_weight2).A

        v_norm_score, mat_ctrl_norm_score = scdrs.method._correct_background(
            v_raw_score=v_raw_score_geneset,
            mat_ctrl_raw_score=mat_ctrl_raw_score,
            v_var_ratio_c2t=v_var_ratio_c2t,
            save_intermediate=None,
        )


        # Get p-values
        mc_p = (1 + (mat_ctrl_norm_score.T >= v_norm_score).sum(axis=0)) / (1 + n_ctrl)
        pooled_p = scdrs.method._get_p_from_empi_null(v_norm_score, mat_ctrl_norm_score.flatten())
        nlog10_pooled_p = -np.log10(pooled_p)
        pooled_z = -sp.stats.norm.ppf(pooled_p).clip(min=-10, max=10)
        
        from statsmodels.stats.multitest import multipletests
        pooled_p_FDR=multipletests(pooled_p, method="fdr_bh")[1]
        nlog10_pooled_p_FDR=-np.log10(pooled_p_FDR)

        dic_res = {
                "raw_score": v_raw_score_geneset,
                "norm_score": v_norm_score,
                "mc_pval": mc_p,
                "pval": pooled_p,
                "nlog10_pval": nlog10_pooled_p,
                "zscore": pooled_z,
                "pval_FDR": pooled_p_FDR,
                "nlog10_pval_FDR": nlog10_pooled_p_FDR
            }
        
        
        if return_ctrl_raw_score:
            for i in range(n_ctrl):
                dic_res["ctrl_raw_score_%d" % i] = mat_ctrl_raw_score[:, i]
        if return_ctrl_norm_score:
            for i in range(n_ctrl):
                dic_res["ctrl_norm_score_%d" % i] = mat_ctrl_norm_score[:, i]

        df_res = pd.DataFrame(index=adata.obs.index, data=dic_res, dtype=np.float32)

        dict_res[group]=df_res
    
    ### Reset the adata.X
    if layer is not None:
        adata.X=X_tmp.copy()
    print("All finished.")
    return(dict_res)





import pandas as pd
def convertTopGeneDictToDF(
    data_dict,
    gene_list_as_string:bool=True):
    """
    Converts the dictionary containing the top genes and their scores reported by `EmerGene` function into a wide-format DataFrame where each condition has two columns:
    "{condition}_Gene" and "{condition}_EG_score".

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are conditions.
        - If `gene_list_as_string=True`: values are "gene:score" formatted strings.
        - If `gene_list_as_string=False`: values are DataFrames with 'Gene' and 'EG_score' columns.

    gene_list_as_string : bool, optional (default=True)
        - If True, assumes values in `data_dict` are strings formatted as "gene:score,gene2:score2,...".
        - If False, assumes values in `data_dict` are DataFrames with 'Gene' and 'EG_score' columns.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each condition has two columns: "{condition}_Gene" and "{condition}_EG_score".
    """
    wide_df = pd.DataFrame()

    for condition, data in data_dict.items():
        if gene_list_as_string:
            # Convert "gene:score" string into a DataFrame
            gene_score_pairs = [pair.split(":") for pair in data.split(",")]
            condition_df = pd.DataFrame(gene_score_pairs, columns=[f"{condition}_Gene", f"{condition}_EG_score"])
            
            # Convert score column to float
            condition_df[f"{condition}_EG_score"] = condition_df[f"{condition}_EG_score"].astype(float)
        else:
            # Assume data is already a DataFrame with 'Gene' and 'EG_score' columns
            condition_df = data.rename(columns={"Gene": f"{condition}_Gene", "EG_score": f"{condition}_EG_score"})

        # Concatenate to the main DataFrame
        if wide_df.empty:
            wide_df = condition_df
        else:
            wide_df = pd.concat([wide_df, condition_df], axis=1)

    return wide_df
      
def runMarkG(adata,
            use_rep:str='X_pca',
             layer:str='log1p',
             n_nearest_neighbors:int=10,
             random_seed:int=27,
             n_repeats:int=3,
             
             mu:float=1,
             sigma:float=100,
            
             remove_lowly_expressed=True,
             expressed_pct=0.1,
    
            
            ):
    
    cellxcell=_build_adjacency_matrix(adata,
                   use_rep=use_rep,
                   n_nearest_neighbors=n_nearest_neighbors,
                       sigma=sigma,
                  )
    
    
    genexcell=adata.layers[layer].T
    
    order1=genexcell @ cellxcell.T
    
    gsp=_calculate_rowwise_cosine_similarity(genexcell, order1)
    
    
    ### Build the random background by sampling
    random_gsp_list=_generate_random_background(adata, cellxcell, genexcell,
                        n_nearest_neighbors=n_nearest_neighbors,
                        n_repeats=n_repeats,
                        random_seed=random_seed)
    
    
    random_gsp=np.mean(random_gsp_list,axis=0)
    
    
    
    adata.var['MarkG_Target']=np.array(gsp).ravel()
    adata.var['MarkG_Random']=np.array(random_gsp).ravel()
    
    
    gsp_score=gsp-mu*random_gsp
    gsp_score=np.array(gsp_score).ravel()
    
    
    if remove_lowly_expressed:
        ### Remaining to be written
        
        
        
        
        
        adata.var['MarkG']=gsp_score
        # adata.var['MarkG'][~adata.var['knn_highly_expressed'].values]=min(gsp_score)-1
        
        
    else:
        adata.var['MarkG']=gsp_score
    print ('The MarkG score for all the genes are save in `MarkG` in `adata.var`')

    

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


import numpy as np
from sklearn.neighbors import KDTree
from scipy import sparse

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
    pooled_p = _get_p_from_empi_null(cellxgene_query, cellxgene_ctrl.A.flatten())
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
          
            
        
    


### Identify gene module
def identifyGeneModule(
    adata,
    gene_list,
    use_rep:str='X_pca',
    resolution:float=0.5,
    n_components:int=30,
    verbosity:int=0
):

    sc.settings.verbosity=verbosity
    adata_gene=sc.AnnData(adata[:,gene_list].X.T)
    adata_gene.obs_names=gene_list
    adata_gene.var_names=adata.obs_names.copy()
    
    ### X_pca
    if use_rep=='X_pca':
        from sklearn.preprocessing import StandardScaler
        expr=StandardScaler(with_mean=False).fit_transform(adata_gene.X)
        from sklearn.decomposition import TruncatedSVD
        transformer = TruncatedSVD(n_components=n_components, random_state=10)
        adata_gene.obsm['X_pca']= transformer.fit_transform(expr)
        ### Run clustering
        sc.pp.neighbors(adata_gene,
                    use_rep='X_pca',
                   n_neighbors=5,random_state=10,knn=True,
                    method="umap")
    elif use_rep=='X':
        sc.pp.neighbors(adata_gene,
                    use_rep='X',
                        metric='cosine',
                   n_neighbors=5,random_state=10,knn=True,
                    method="umap")
        
    sc.tl.leiden(adata_gene,resolution=resolution,key_added='Module')
    adata_gene.obs['Module']='Module'+adata_gene.obs['Module'].astype(str)
    
    ### Order the modules by similarity
    adata_gene.obs['Module']=adata_gene.obs['Module'].astype('category')
    if use_rep=='X_pca':
        sc.tl.dendrogram(adata_gene, groupby='Module', use_rep=use_rep)
    
    return (adata_gene.obs)

### Refer to Scanpy for _select_top_n function
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices

### Normalization based on information
def infog(
    adata,
    copy:bool=False,
    layer='raw',
    n_top_genes:int=1000,
    key_added:str='infog',
    random_state:int =10,
    trim:bool=True,
    verbosity:int=1
):
    
    adata = adata.copy() if copy else adata
    
    if sparse.issparse(adata.layers[layer]):
        counts=adata.layers[layer]
    else:
        counts=sparse.csr_matrix(adata.layers[layer])
    

    cell_depth=counts.sum(axis=1).A
    gene_depth=counts.sum(axis=0).A   
    counts_sum  = np.sum(counts)
    scale = np.median(cell_depth.ravel())
    ### should use this one, especially for downsampling experiment, only this one works, the sequencing baises are corrected, partially because only this transformation is linear
    normalized =  sparse.diags(scale/cell_depth.ravel()) @ counts 
    
    info_factor=sparse.diags(counts_sum/cell_depth.ravel()) @ counts @ sparse.diags(1/gene_depth.ravel())
    normalized2=normalized.multiply(info_factor).sqrt()
    if trim:
        threshold = np.sqrt(counts.shape[0])
        normalized2.data[normalized2.data >  threshold] =  threshold

    adata.layers[key_added]=normalized2
    
    ### Calculate the variance
    c = normalized2.copy()
    mean=np.array(c.mean(axis=0))
    mean **=2
    c.data **= 2
    residual_var_orig_b=np.squeeze(np.array(c.mean(axis=0))-mean) 
#     del c
    adata.var[key_added+'_var']=residual_var_orig_b
    
    
    ### Feature selection    
    pos_gene=_select_top_n(adata.var[key_added+'_var'],n_top_genes)
    tmp=np.repeat(False,adata.n_vars)
    tmp[pos_gene]=True
    adata.var['highly_variable_'+key_added]=tmp
    if verbosity>0:
        print(f'The normalized data is saved as `{key_added}` in `adata.layers`.')
        print(f'The highly variable genes are saved as `highly_variable_{key_added}` in `adata.obs`.')
        print(f'Finished INFOG normalization.')
         
    ### Return the result
    return adata if copy else None
    