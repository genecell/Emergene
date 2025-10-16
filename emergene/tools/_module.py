### Identify gene module
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

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