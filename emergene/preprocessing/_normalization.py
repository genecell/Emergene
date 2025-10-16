### Normalization based on information
import numpy as np
from scipy import sparse
from anndata import AnnData
from typing import Optional


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



### Refer to Scanpy for _select_top_n function
def _select_top_n(scores, n_top):
    reference_indices = np.arange(scores.shape[0], dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]
    return global_indices
