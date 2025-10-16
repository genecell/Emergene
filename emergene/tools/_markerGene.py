import numpy as np
from anndata import AnnData

from ._utils import _build_adjacency_matrix
from ._utils import (
    _calculate_rowwise_cosine_similarity,
    _generate_random_background,
)

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