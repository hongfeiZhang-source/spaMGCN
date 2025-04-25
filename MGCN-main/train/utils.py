import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca
#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.001, end=7.0, increment=0.001, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
# def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
#     '''\
#     Searching corresponding resolution according to given cluster number
    
#     Parameters
#     ----------
#     adata : anndata
#         AnnData object of spatial data.
#     n_clusters : int
#         Targetting number of clusters.
#     method : string
#         Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
#     use_rep : string
#         The indicated representation for clustering.
#     start : float
#         The start value for searching.
#     end : float 
#         The end value for searching.
#     increment : float
#         The step size to increase.
        
#     Returns
#     -------
#     res : float
#         Resolution.
        
#     '''
#     print('Searching resolution...')
#     label = 0
#     sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
#     for res in sorted(list(np.arange(start, end, increment)), reverse=True):
#         if method == 'leiden':
#            sc.tl.leiden(adata, random_state=0, resolution=res)
#            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
#            print('resolution={}, cluster number={}'.format(res, count_unique))
#         elif method == 'louvain':
#            sc.tl.louvain(adata, random_state=0, resolution=res)
#            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
#            print('resolution={}, cluster number={}'.format(res, count_unique))
#         if count_unique == n_clusters:
#             label = 1
#             break

#     assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
#     return res 
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01, max_iters=100):
    '''\
    Searching corresponding resolution according to given cluster number using binary search.
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Target number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    tol : float
        Tolerance for stopping the search. The search stops when the interval size is smaller than `tol`.
    max_iters : int
        Maximum number of iterations for the binary search.
        
    Returns
    -------
    res : float
        Resolution that produces the target number of clusters.
        
    Raises
    ------
    ValueError
        If the target number of clusters cannot be found within the given range.
    '''
    print('Searching resolution using binary search...')
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    def get_cluster_count(res):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            return len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            return len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    # Binary search
    left, right = start, end
    for i in range(max_iters):
        mid = (left + right) / 2
        count = get_cluster_count(mid)
        print(f'resolution={mid:.4f}, cluster number={count}')
        
        if count == n_clusters:
            return mid
        elif count < n_clusters:
            left = mid  # Increase resolution
        else:
            right = mid  # Decrease resolution
        
        # Check if the interval is small enough
        if (right - left) < increment:
            break
    
    # Final check within the tolerance range
    final_res = (left + right) / 2
    final_count = get_cluster_count(final_res)
    if final_count == n_clusters:
        return final_res
    else:
        raise ValueError(f"Resolution not found for {n_clusters} clusters. Try a wider range or smaller tolerance.")    
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01, max_iters=100):
    '''\
    Searching corresponding resolution according to given cluster number using binary search.
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Target number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The increment value for searching around the found resolution.
    max_iters : int
        Maximum number of iterations for the binary search.
        
    Returns
    -------
    res : float
        Resolution that produces the target number of clusters.
        
    Raises
    ------
    ValueError
        If the target number of clusters cannot be found within the given range.
    '''
    print('Searching resolution using binary search...')
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    def get_cluster_count(res):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            return len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            return len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    # Binary search
    left, right = start, end
    found_res = None
    for i in range(max_iters):
        mid = (left + right) / 2
        count = get_cluster_count(mid)
        print(f'resolution={mid:.4f}, cluster number={count}')
        
        if count == n_clusters:
            found_res = mid
            break
        elif count < n_clusters:
            left = mid  # Increase resolution
        else:
            right = mid  # Decrease resolution
        
        # Check if the interval is small enough
        if (right - left) < increment:
            break
    
    if found_res is None:
        raise ValueError(f"Resolution not found for {n_clusters} clusters. Try a wider range or smaller tolerance.")
    
    # Search around the found resolution
    res_list = [found_res]
    for i in range(1, 3):
        res_forward = found_res + i * increment
        res_backward = found_res - i * increment
        
        if res_backward >= start:
            count_backward = get_cluster_count(res_backward)
            if count_backward == n_clusters:
                res_list.append(res_backward)
        
        if res_forward <= end:
            count_forward = get_cluster_count(res_forward)
            if count_forward == n_clusters:
                res_list.append(res_forward)
    
    # Calculate mean of the resolutions
    mean_res = sum(res_list) / len(res_list)
    mean_count = get_cluster_count(mean_res)
    
    if mean_count == n_clusters:
        return mean_res
    else:
        # If mean_res does not work, return the median of the resolutions
        median_res = sorted(res_list)[len(res_list) // 2]
        median_count = get_cluster_count(median_res)
        if median_count == n_clusters:
            return median_res
        # else:
        #     raise ValueError(f"Resolution not found for {n_clusters} clusters. Try a wider range)

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     
