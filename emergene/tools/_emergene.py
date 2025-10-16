"""
Main emergene algorithm for individual cell-based differential transcriptomic analysis across conditions.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy import sparse
from sklearn.preprocessing import normalize
from typing import Union, Optional, Dict, Tuple
from anndata import AnnData
import warnings

# Import helper functions from ._utils
from ._utils import (
    _build_adjacency_matrix,
    _build_adjacency_matrix_acrossDataset,
    _calculate_rowwise_cosine_similarity,
    _generate_random_background,
    _select_top_n
)


def runEMERGENE(
    adata: AnnData,
    use_rep: str = 'X_pca',
    use_rep_acrossDataset: str = 'X_pca',
    layer: Optional[str] = None,
    n_nearest_neighbors: int = 10,
    condition_key: str = 'Sample',
    random_seed: int = 27,
    n_repeats: int = 3,
    mu: float = 1.0,
    beta: float = 1.0,
    sigma: float = 100.0,
    n_cells_expressed_threshold: int = 50,
    n_top_EG_genes: int = 500,
    remove_lowly_expressed: bool = True,
    expressed_pct: float = 0.1,
    inplace: bool = False,
    gene_list_as_string: bool = False,
    verbose: int = 1,
) -> Union[Tuple[Dict[str, Union[str, pd.DataFrame]], pd.DataFrame], 
           Dict[str, Union[str, pd.DataFrame]]]:
    """
    Identify emergent genes across different biological conditions.
    
    emergene identifies genes that show coordinated local expression patterns within
    specific conditions by using graph-based diffusion and cosine similarity. The 
    method compares gene expression patterns within a condition against random 
    backgrounds and other conditions to identify truly emergent patterns.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing preprocessed single-cell RNA-seq data.
        Should contain low-dimensional representations (e.g., PCA) in `.obsm`.
    use_rep : str, default='X_pca'
        Key in `adata.obsm` for the low-dimensional embedding used for 
        condition-specific diffusion. Common choices include 'X_pca', 'X_umap',
        or other dimensionality reduction results.
    use_rep_acrossDataset : str, default='X_pca'
        Key in `adata.obsm` for computing the across-dataset connectivity matrix
        using bbknn. Can be the same as `use_rep` or a different representation.
    layer : str or None, default=None
        Key in `adata.layers` for the gene expression matrix to use. If `None`,
        uses the default expression matrix stored in `adata.X`. Recommended to
        use log-normalized counts (e.g., 'log1p') or infog-normalized data.
    n_nearest_neighbors : int, default=10
        Number of nearest neighbors used when constructing adjacency matrices.
        Higher values increase smoothing but may dilute local patterns.
    condition_key : str, default='Sample'
        Column name in `adata.obs` that specifies the condition or batch label
        for each cell. Must be a categorical variable.
    random_seed : int, default=27
        Seed for the random number generator to ensure reproducibility of
        randomization procedures.
    n_repeats : int, default=3
        Number of randomizations to perform for background generation. Higher
        values provide more stable background estimates but increase computation.
        Minimum recommended: 3, typical: 10-30.
    mu : float, default=1.0
        Weight for subtracting the random background specificity in the final
        emergene score. Set to 0 to disable random background correction.
    beta : float, default=1.0
        Weight for subtracting the condition-wise background specificity in the
        final emergene score. Set to 0 to disable cross-condition correction.
    sigma : float, default=100.0
        Scaling parameter for the exponential kernel in adjacency matrix
        construction. Larger values result in slower decay of edge weights.
    n_cells_expressed_threshold : int, default=50
        Minimum number of cells in which a gene must be expressed to be
        considered. Genes below this threshold receive minimum scores.
    n_top_EG_genes : int, default=500
        Number of top emergent genes to select for each condition based on
        emergene scores.
    remove_lowly_expressed : bool, default=True
        Flag indicating whether to filter lowly expressed genes. Currently
        implemented via `n_cells_expressed_threshold`.
    expressed_pct : float, default=0.1
        Minimum percentage of cells in which a gene must be expressed. 
        Note: Currently not implemented, planned for future versions.
    inplace : bool, default=False
        If True, saves emergene scores directly into `adata.var` and modifies
        the AnnData object in-place. If False, returns scores as a DataFrame.
    gene_list_as_string : bool, default=False
        If True, saves top genes and scores as a comma-separated string in the
        format "gene1:score1,gene2:score2,...". If False, saves as a pandas
        DataFrame with separate columns for genes and scores.
    verbose : int, default=1
        Verbosity level. 0: silent, 1: progress messages, 2: detailed output.
    
    Returns
    -------
    If `inplace=False`:
        Tuple[Dict, pd.DataFrame]
            - Dictionary where keys are f'EG_{condition}' and values are either
              strings (if `gene_list_as_string=True`) or DataFrames containing
              the top emergent genes and their scores for each condition.
            - DataFrame with columns f'EmerGene_{condition}' containing emergene
              scores for all genes across all conditions.
    
    If `inplace=True`:
        Dict
            Dictionary of top emergent genes. The AnnData object is modified
            in-place with emergene scores added to `adata.var` and local fold
            changes added to `adata.layers['localFC']`.
    
    Notes
    -----
    The method computes three components for each gene:
    
    1. Target specificity (GSP): Cosine similarity between original expression
       and diffused expression within the condition
    2. Random background: Average GSP from randomly permuted adjacency matrices
    3. Condition-wise background: GSP between target condition and other conditions
    
    The final emergene score is: GSP - μ × random_GSP - β × condition_GSP
    
    The local fold change matrix is stored in `adata.layers['localFC']` and
    represents log1p-transformed fold changes of each gene in each cell relative
    to the cross-condition background.
    
    Raises
    ------
    ValueError
        If `use_rep` or `use_rep_acrossDataset` is not found in `adata.obsm`.
        If `condition_key` is not found in `adata.obs`.
        If `layer` is specified but not found in `adata.layers`.
        If numeric parameters are out of acceptable ranges.
    ImportError
        If required dependency `bbknn` is not installed.
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> import scanpy as sc
    >>> import emergene as eg
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> gene_dict, scores = eg.tl.runEMERGENE(adata, condition_key='cell_type')
    >>> print(gene_dict['EG_T_cell'].head())
    
    Using custom parameters and saving in-place:
    
    >>> gene_dict = eg.tl.runEMERGENE(
    ...     adata,
    ...     condition_key='treatment',
    ...     n_top_EG_genes=1000,
    ...     mu=1.5,
    ...     beta=0.5,
    ...     inplace=True
    ... )
    >>> print(adata.var['EmerGene_treated'].head())
    
    See Also
    --------
    runMarkG : Marker gene identification without condition comparison
    infog : INFOG normalization for preprocessing
    score : Gene set enrichment scoring
    
    References
    ----------
    .. [1] Wu et al., Pyramidal neurons proportionately alter the identity and survival of specific cortical interneuron subtypes, bioRxiv (2024).
    """
    
    # ==================== Input Validation ====================
    if use_rep not in adata.obsm:
        raise ValueError(
            f"'{use_rep}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    if use_rep_acrossDataset not in adata.obsm:
        raise ValueError(
            f"'{use_rep_acrossDataset}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    if condition_key not in adata.obs:
        raise ValueError(
            f"'{condition_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.keys())}"
        )
    
    if layer is not None and layer not in adata.layers:
        raise ValueError(
            f"'{layer}' not found in adata.layers. "
            f"Available keys: {list(adata.layers.keys())}"
        )
    
    # Check for required packages
    try:
        import scanpy as sc
        if not hasattr(sc.external.pp, 'bbknn'):
            raise ImportError("bbknn not found in scanpy.external")
    except ImportError as e:
        raise ImportError(
            "bbknn is required for runEMERGENE. "
            "Install with: pip install bbknn"
        ) from e
    
    # Validate numeric parameters
    if n_nearest_neighbors < 1:
        raise ValueError(f"n_nearest_neighbors must be >= 1, got {n_nearest_neighbors}")
    
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    
    if n_top_EG_genes < 1:
        raise ValueError(f"n_top_EG_genes must be >= 1, got {n_top_EG_genes}")
    
    if n_cells_expressed_threshold < 0:
        raise ValueError(
            f"n_cells_expressed_threshold must be >= 0, "
            f"got {n_cells_expressed_threshold}"
        )
    
    # Warnings for common issues
    condition_info = adata.obs[condition_key]
    conditions = np.unique(condition_info)
    n_conditions = len(conditions)
    
    if n_conditions < 2:
        warnings.warn(
            f"Only {n_conditions} condition found in '{condition_key}'. "
            "runEMERGENE is designed for comparing multiple conditions. "
            "Consider using runMarkG() instead for single-condition analysis.",
            UserWarning
        )
    
    if verbose > 0:
        print(f"emergene v1.0.0 - runEMERGENE")
        print(f"=" * 60)
        print(f"Number of cells: {adata.n_obs}")
        print(f"Number of genes: {adata.n_vars}")
        print(f"Number of conditions: {n_conditions}")
        print(f"Conditions: {', '.join(map(str, conditions))}")
        print(f"Parameters:")
        print(f"  - n_neighbors: {n_nearest_neighbors}")
        print(f"  - mu: {mu}, beta: {beta}, sigma: {sigma}")
        print(f"  - n_repeats: {n_repeats}")
        print(f"  - n_top_EG_genes: {n_top_EG_genes}")
        print(f"=" * 60)
        print("")
    
    # ==================== Main Analysis ====================
    
    # Extract gene expression data from the specified layer or default to adata.X
    if layer is None:
        genexcell = adata.X.T
    else:
        genexcell = adata.layers[layer].T
    
    # Initialize output structures
    EG_dict = {}  # Dictionary for condition-wise DataFrames
    
    # If not inplace, initialize a DataFrame to store emergene scores
    if not inplace:
        emergene_scores = pd.DataFrame(index=adata.var_names)
    
    # Calculate the number of expressed cells for each gene
    if issparse(adata.X):
        n_cells_expressed = adata.X.getnnz(axis=0)
    else:
        n_cells_expressed = np.sum(adata.X > 0, axis=0)
    
    # Create lists to store fold change matrices and indices
    fold_change_list = []
    index_list = []
    
    # Build cross-dataset adjacency matrix once (used for all conditions)
    if verbose > 0:
        print("Building cross-dataset adjacency matrix...")
    
    cellxcell_acrossDataset = _build_adjacency_matrix_acrossDataset(
        adata, 
        use_rep=use_rep_acrossDataset, 
        condition_key=condition_key, 
        n_nearest_neighbors=n_nearest_neighbors
    )
    
    # ==================== Process Each Condition ====================
    for condition in conditions:
        if verbose > 0:
            print(f"\nProcessing condition: {condition}")
            print(f"-" * 60)
        
        # Get indices for target condition
        idx_cells = np.where(condition_info == condition)[0]
        
        # Build adjacency matrix for target condition
        if verbose > 1:
            print(f"  Building adjacency matrix for {len(idx_cells)} cells...")
        
        cellxcell_target = _build_adjacency_matrix(
            adata[condition_info == condition],
            use_rep=use_rep,
            n_nearest_neighbors=n_nearest_neighbors,
            sigma=sigma,
        )
        
        # Extract gene expression for target condition
        genexcell_target = genexcell[:, idx_cells]
        
        # Calculate random background specificity
        if verbose > 1:
            print(f"  Generating random background ({n_repeats} repeats)...")
        
        random_gsp_list = _generate_random_background(
            adata[condition_info == condition], 
            cellxcell_target, 
            genexcell_target,
            n_nearest_neighbors=n_nearest_neighbors,
            n_repeats=n_repeats,
            random_seed=random_seed
        )
        random_gsp = np.mean(random_gsp_list, axis=0)
        
        # Normalize adjacency matrix (L1 normalization)
        cellxcell_target = normalize(cellxcell_target, axis=1, norm='l1')
        
        # Diffuse expression on the same dataset
        if verbose > 1:
            print(f"  Computing target specificity...")
        
        genexcell_target_diffusion = genexcell_target @ cellxcell_target.T
        gsp = _calculate_rowwise_cosine_similarity(
            genexcell_target, 
            genexcell_target_diffusion
        )
        
        # Calculate condition-wise specificity
        if verbose > 1:
            print(f"  Computing cross-condition specificity...")
        
        idx_remaining_cells = np.where(condition_info != condition)[0]
        cellxcell_targetXremaining = cellxcell_acrossDataset[idx_cells, :][:, idx_remaining_cells]
        
        # Normalize cross-condition adjacency matrix
        cellxcell_targetXremaining = normalize(cellxcell_targetXremaining, axis=1, norm='l1')
        
        genexcell_remaining = genexcell[:, idx_remaining_cells]
        genexcell_conditionwise_diffusion = genexcell_remaining @ cellxcell_targetXremaining.T
        
        gsp_conditionWise = _calculate_rowwise_cosine_similarity(
            genexcell_target, 
            genexcell_conditionwise_diffusion
        )
        
        # Calculate local fold change
        if verbose > 1:
            print(f"  Computing local fold changes...")
        
        inv_b = genexcell_conditionwise_diffusion.copy()
        inv_b.data = 1.0 / inv_b.data - 1.0
        
        fold_change = genexcell_target_diffusion.multiply(inv_b) + genexcell_target_diffusion
        fold_change_list.append(fold_change)
        index_list.append(idx_cells)
        
        # Compute final emergene score
        gsp_score = gsp - mu * random_gsp - beta * gsp_conditionWise
        gsp_score = np.array(gsp_score).ravel()
        
        # Filter by number of expressed cells
        gsp_score[n_cells_expressed < n_cells_expressed_threshold] = gsp_score.min() - 1e-6
        
        # Select top genes
        if verbose > 1:
            print(f"  Selecting top {n_top_EG_genes} genes...")
        
        EG_genes_idx = _select_top_n(gsp_score, n_top_EG_genes)
        EG_genes_list = adata.var_names.values[EG_genes_idx]
        EG_genes_list_scores = np.round(gsp_score[EG_genes_idx], 6)
        
        # Store results
        if gene_list_as_string:
            EG_dict[f'EG_{condition}'] = ",".join(
                f"{gene}:{score:.6f}" 
                for gene, score in zip(EG_genes_list, EG_genes_list_scores)
            )
        else:
            EG_dict[f'EG_{condition}'] = pd.DataFrame({
                'Gene': EG_genes_list, 
                'EG_score': EG_genes_list_scores
            })
        
        # Save emergene scores
        if inplace:
            adata.var[f'EmerGene_{condition}'] = gsp_score
            if verbose > 0:
                print(f"  ✓ Scores saved in adata.var['EmerGene_{condition}']")
        else:
            emergene_scores[f'EmerGene_{condition}'] = gsp_score
            if verbose > 0:
                print(f"  ✓ Scores added to output DataFrame")
    
    # ==================== Save Local Fold Changes ====================
    if verbose > 0:
        print(f"\n{'=' * 60}")
        print("Finalizing results...")
    
    # Merge and reorder fold change matrices
    original_indices = np.hstack(index_list)
    new_indices = range(len(original_indices))
    sorted_dict = dict(sorted(dict(zip(original_indices, new_indices)).items()))
    reorder_indices = list(sorted_dict.values())
    
    merged_sparse_matrix = sparse.hstack(fold_change_list)
    merged_sparse_matrix = merged_sparse_matrix[:, reorder_indices].T
    merged_sparse_matrix.data = np.log1p(merged_sparse_matrix.data)
    
    adata.layers['localFC'] = merged_sparse_matrix
    
    if verbose > 0:
        print(f"✓ Local fold changes saved in adata.layers['localFC']")
        print(f"✓ Analysis complete!")
        print(f"{'=' * 60}\n")
    
    # Return results
    if not inplace:
        return EG_dict, emergene_scores
    else:
        return EG_dict