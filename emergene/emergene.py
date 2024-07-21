import numpy as np
from scipy.sparse import csr_matrix, issparse
import scanpy as sc

def buildADJ(adata,
            use_rep:str='X_pca',
             n_nearest_neighbors:int=10,
             
             
            ):
    
    X_rep=adata.obsm[use_rep].copy()
    from sklearn.neighbors import kneighbors_graph
    cellxcell = kneighbors_graph(X_rep,
                         n_neighbors=n_nearest_neighbors,
                         mode='distance', include_self=False)
    
    cellxcell.data=1/np.exp(cellxcell.data/100)
    
    return cellxcell


import numpy as np
from scipy.sparse import issparse

def rowwise_cosine_similarityV2(A, B):
    """
    Computes the cosine similarity between corresponding rows of matrices A and B.

    Parameters:
    - A (numpy.ndarray or scipy.sparse.csr_matrix): A 2D array or CSR sparse matrix representing matrix A with shape (n, m).
    - B (numpy.ndarray or scipy.sparse.csr_matrix): A 2D array or CSR sparse matrix representing matrix B with shape (n, m).

    Returns:
    - cosine_similarities (numpy.ndarray): A 1D array of shape (n,) containing the cosine similarity between corresponding rows of A and B.
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
    # cosine_similarities = np.where(denominator!=0, dot_products / denominator, 0)
    cosine_similarities = np.divide(dot_products, denominator, out=np.zeros_like(dot_products), where=denominator!=0)
    
    return cosine_similarities



from scipy.sparse import csr_matrix
### To efficiently create a random adjacency matrix
def buildRandomADJ(
    adata,
    cellxcell,
    n_nearest_neighbors:int=10,
    random_seed:int=0,
             
             
            ):
    
    row_ind=np.repeat(np.arange(adata.n_obs), n_nearest_neighbors)
    np.random.seed(random_seed)
    col_ind=np.random.choice(np.arange(adata.n_obs), adata.n_obs*n_nearest_neighbors, replace=True)
    
    ### Shuffle the weights
    connectivity=cellxcell.data.copy()
    np.random.shuffle(connectivity)

    cellxcell_shuffle=csr_matrix((connectivity,(row_ind, col_ind)),shape=cellxcell.shape)
    
    return cellxcell_shuffle



def generateRandomBackground(
    adata,
    cellxcell,
    genexcell,
    n_nearest_neighbors:int=10,
    n_repeats:int=30,
    random_seed:int=0,
    
):
    random_gsp_list=[]
    np.random.seed(random_seed)
    
    random_seed_list=np.random.choice(1902772, size=n_repeats, replace=False)
    # print(random_seed_list)
    
    for i in np.arange(n_repeats):
        random_seed=random_seed_list[i]
    
        cellxcell_shuffle=buildRandomADJ(adata,
                                         cellxcell,
                                         n_nearest_neighbors=n_nearest_neighbors,
                                         random_seed=random_seed)
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_shuffle=normalize(cellxcell_shuffle, axis=1, norm='l1')
        
    
        random_order1=genexcell @ cellxcell_shuffle.T


        random_gsp_list.append(rowwise_cosine_similarityV2(genexcell, random_order1))
    
    return(random_gsp_list)


def buildADJ_acrossDataset(adata,
            use_rep:str='X_pca',
                           condition_key:str='Sample',
             n_nearest_neighbors:int=10,
             
             
            ):
    
    X_rep=adata.obsm[use_rep].copy()
    
    adata_copy=sc.external.pp.bbknn(adata, batch_key=condition_key, use_rep=use_rep, copy=True, neighbors_within_batch=n_nearest_neighbors)
    cellxcell_acrossDataset=adata_copy.obsp['connectivities']
    
    return cellxcell_acrossDataset






from scipy import sparse
from sklearn.preprocessing import normalize
def EmerGene(
    adata,
    use_rep:str='X_pca',
    use_rep_acrossDataset:str='X_pca',
    layer:str='log1p',
    n_nearest_neighbors:int=10,
    condition_key:str='Sample',
    random_seed:int=27,
    n_repeats:int=3,

    mu:float=1,
    beta:float=1,

    remove_lowly_expressed=True,
    expressed_pct=0.1,

):
    
    
    
    ### Record the sample info
    sample_info=adata.obs[condition_key]
    samples=np.unique(sample_info)
    n_samples=len(samples)
    
    genexcell=adata.layers[layer].T
    
    # Create a copy of genexcell to modify
    # genexcell_foldchange = genexcell.A.copy()
    ### Create a list to store all the sparse matrix, and then reorder it later
    fold_change_list=list()
    index_list=list()
    
    
    cellxcell_acrossDataset=buildADJ_acrossDataset(adata, use_rep=use_rep_acrossDataset, condition_key=condition_key, n_nearest_neighbors=n_nearest_neighbors)
    
    
    for sample in samples:
        print('processing sample: ', sample)
        ### get the info of the target sample
        idx_cells = np.where(sample_info == sample)[0]

        cellxcell_target=buildADJ(adata[sample_info == sample],
                       use_rep=use_rep,
                       n_nearest_neighbors=n_nearest_neighbors
                      )
        
        
        ### Calculate the target sample specificity
        genexcell_target=genexcell[:,idx_cells]
        
        ### Calculate the target sample's background specificity
        ### Build the random background by sampling
        random_gsp_list=generateRandomBackground(adata[sample_info == sample], cellxcell_target, genexcell_target,
                            n_nearest_neighbors=n_nearest_neighbors,
                            n_repeats=n_repeats,
                            random_seed=random_seed)
        random_gsp=np.mean(random_gsp_list,axis=0)
        
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_target=normalize(cellxcell_target, axis=1, norm='l1')
        ### Diffused on the same dataset
        genexcell_target_diffusion=genexcell_target @ cellxcell_target.T
       
        gsp=rowwise_cosine_similarityV2(genexcell_target, genexcell_target_diffusion)

        


        ### Calculate the sample-wise specificty
        idx_remaining_cells = np.where(sample_info != sample)[0]
        cellxcell_targetXremaining=cellxcell_acrossDataset[idx_cells,:][:,idx_remaining_cells]
        
        ### Need to normalize the weights, then the add up would be comparable, l1 normalization for each row
        cellxcell_targetXremaining=normalize(cellxcell_targetXremaining, axis=1, norm='l1')

        genexcell_remaining=genexcell[:,idx_remaining_cells]
        
        genexcell_samplewise_diffusion=genexcell_remaining @ cellxcell_targetXremaining.T
        
        gsp_sampleWise=rowwise_cosine_similarityV2(genexcell_target, genexcell_samplewise_diffusion)
        
        
        ### Record the fold change
        # Create a copy of B to avoid modifying the original matrix
        inv_b = genexcell_samplewise_diffusion.copy()
        # Invert non-zero elements safely. Zeros in B remain zero in inv_b, which acts like ones in division. Then minus 1.0, this is used to combine with add genexcell_target itself, then for the values nonzero in A, but zero in B, keep it unchanged in A during the multiplication
        inv_b.data = 1.0 / inv_b.data -1.0
        # Reset the genexcell_foldchange, and will return it back after the calculation
        # genexcell_foldchange[:,idx_cells] = 
        ## use genexcell_target_diffusion instead of genexcell_target, to represent the local average expression
        fold_change_list.append(genexcell_target_diffusion.multiply(inv_b) + genexcell_target_diffusion)
        index_list.append(idx_cells)

        gsp_score=gsp-mu*random_gsp-beta*gsp_sampleWise
        gsp_score=np.array(gsp_score).ravel()

        adata.var['EmerGene_'+sample]=gsp_score
        
        
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
    print("The local foldchange information are saved in `localFC` in adata.layers.")
      


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
    random_seed:int=27
):
    ### Could replace this with the idea of ChromVAR or pyChromVAR, to construct a gene graph, then the sampling of genes would also be easy
    ### I used the sampling of genes, when I do the GWAS enrich analysis, please double check that code as well
    if layer is not None:
        X_tmp=adata.X.copy()
        adata.X=adata.layers[layer]
    scdrs.preprocess(adata, n_mean_bin=n_mean_bin, n_var_bin=n_var_bin, copy=False)

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[adata.var_names].copy()
    df_gene["gene"] = df_gene.index
    df_gene.drop_duplicates(subset="gene", inplace=True)
    
    
    ### To save the results for each geneset
    dict_res=dict()
    print("The list of groups to process: ", list(geneset_dict.keys()))
    for group, gene_and_weight in geneset_dict.items():
        print("Processing :", group)
        
        gene_dict = dict(item.split(":") for item in gene_and_weight.split(","))
        # Convert string values to float
        gene_dict = {k: float(v) for k, v in gene_dict.items()}

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

        dic_res = {
                "raw_score": v_raw_score_geneset,
                "norm_score": v_norm_score,
                "mc_pval": mc_p,
                "pval": pooled_p,
                "nlog10_pval": nlog10_pooled_p,
                "zscore": pooled_z,
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
        adata.X=X_tmp
    print("All finished.")
    return(dict_res)
    

      
def runMarkG(adata,
            use_rep:str='X_pca',
             layer:str='log1p',
             n_nearest_neighbors:int=10,
             random_seed:int=27,
             n_repeats:int=3,
             
             mu:float=1,
            
             remove_lowly_expressed=True,
             expressed_pct=0.1,
    
            
            ):
    
    cellxcell=buildADJ(adata,
                   use_rep=use_rep,
                   n_nearest_neighbors=n_nearest_neighbors
                  )
    
    
    genexcell=adata.layers[layer].T
    
    order1=genexcell @ cellxcell.T
    
    gsp=rowwise_cosine_similarityV2(genexcell, order1)
    
    
    ### Build the random background by sampling
    random_gsp_list=generateRandomBackground(adata, cellxcell, genexcell,
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

    
### Refer to Scanpy
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices

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
    