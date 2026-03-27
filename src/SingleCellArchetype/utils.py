"""SingleCellArchetype utils
"""

import numpy as np
from scipy.stats import zscore
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from py_pcha import PCHA


def norm(x, depths):
    """
    Arguments: 
        x - cell by gene count matrix
        depths - sequencing depth per cell
        
    Output:
        xn - normalized count matrix

    This function takes raw counts as the input, and does the following steps sequencially.
         1. size normalization (CP10k) 
         2. log1p normalization (base 2 - log2(1+CP10k))
         3. zscore per gene  
    """

    xn = x/depths.reshape(-1,1)*1e4
    xn = np.log2(1+xn)
    xn = zscore(xn, axis=0)

    if np.any(np.isnan(xn)):
        print('Warning: the normalized matrix contains nan values. Check input.')

    return xn

def proj(x_norm, ndim, method='PCA', skip_pc1=False):
    """
    Arguments: 
        x_norm - normalized cell by gene feature matrix
        ndim   - number of dimensions

    Output:
        x_proj - a low-dimensional representation of `x_norm` 

    Here we only implemented PCA - a common projection method widely used, including by
    Adler et al. 2019 and Xie et al. 2024 for the Achetypal Analysis of scRNA-seq data.

    In principle, one can also choose to use other projection methods as needed.
    """

    if method == 'PCA':
        pca_model = PCA(n_components=ndim+1)
        x_proj = pca_model.fit_transform(x_norm)
    else:
        raise ValueError('methods other than PCA are not implemented...')

    # trim (last one or first one)
    if skip_pc1:
        x_proj = x_proj[:,1:]
    else:
        x_proj = x_proj[:,:-1]

    return x_proj, pca_model

def proj_transform(x_norm, pca_model, skip_pc1=False):
    """Project data using an already-fitted PCA model (no re-fitting).

    Arguments:
        x_norm    - normalized cell by gene feature matrix
        pca_model - a fitted sklearn PCA object (from proj())
        skip_pc1  - if True, drop PC1 and keep the next ndim components

    Output:
        x_proj - low-dimensional projection of x_norm
    """
    x_proj = pca_model.transform(x_norm)
    if skip_pc1:
        x_proj = x_proj[:,1:]
    else:
        x_proj = x_proj[:,:-1]
    return x_proj

def pcha(X, noc=3, delta=0, **kwargs):
    """
        X = XCS + err
        
        X  - (ndim, ncell) - original data
        C  - (ncell, noc)  - definition of archetypes as a linear summation of cells
        S  - (noc, ncell)  - cell locations reconstructed by archetype locations

        XC = X.dot(C) - (ndim, noc) - archetype locations
    """
    XC, S, C, SSE, varexpl = PCHA(X, noc=noc, delta=delta, **kwargs)
    XC = np.array(XC)
    XC = XC[:,np.argsort(XC[0])] # assign an order according to x-axis 
    return XC, varexpl 

def downsamp(x, which='cell', p=0.8, seed=None, return_cond=False):
    """
    Arguments:
        x - cell by gene matrix
        which - downsample cells (rows) or genes (columns)
        p - fraction of cells/genes to keep - should be a value between ~ [0,1]
    """
    n0, n1 = x.shape
    
    rng = np.random.default_rng(seed=seed)
    
    if which in [0, 'cell', 'row']:
        cond = rng.random(n0)<p
        xout = x[cond, :]    
    elif which in [1, 'gene', 'col', 'column']:
        cond = rng.random(n1)<p
        xout = x[:, cond]
    else:
        raise ValueError('choose from cell or gene')
    
    if return_cond:
        return xout, cond
    else:
        return xout

def bootstrap(x, which='cell', seed=None, return_cond=False):
    """
    Arguments:
        x - cell by gene matrix
        which - bootstrap cells (rows) or genes (columns)

        bootstrap is downsample with replacement using the same cell / gene size
    """
    n0, n1 = x.shape
    
    rng = np.random.default_rng(seed=seed)

    if which in [0, 'cell', 'row']:
        idx = rng.choice(n0, size=n0, replace=True)
        xout = x[idx, :]
    elif which in [1, 'gene', 'col', 'column']:
        idx = rng.choice(n1, size=n1, replace=True)
        xout = x[:, idx]
    else:
        raise ValueError('choose from cell or gene')
    
    if return_cond:
        return xout, idx 
    else:
        return xout

def bootstrap_or_downsamp(x, which='cell', 
                          is_bootstrap=True, downsamp_p=None, 
                          seed=None, return_cond=False):
    """
    choose bootstrap (with replacement)
    or downsample with a specified proportion (without replacement)
    """

    if is_bootstrap:
        # bootstrap mode
        return bootstrap(x, which='cell', seed=seed, return_cond=return_cond)

    elif downsamp_p is not None:
        # downsamp mode
        if not (0 < downsamp_p < 1):
            raise ValueError(f"downsamp_p must be in (0, 1), got {downsamp_p}")

        return downsamp(x, which='cell', p=downsamp_p, seed=seed, return_cond=return_cond)
    else:
        raise ValueError("choose Bootstrap or Downsamp")

def shuffle_rows_per_col(x, seed=None):
    """
    Arguments:
       x - cell by gene matrix
       seed - a random seed for reproducibility
    
    shuffles entries across rows (cells) independently for each col (gene)
    """
    rng = np.random.default_rng(seed=seed)
    x_shuff = rng.permuted(x, axis=0)
    return x_shuff

def get_t_ratio(xp, aa):
    """
    Arguments:
        xp -- projected matrix (ncell by ndim)
        aa -- inferred archetypes (ndim by noc)
     
    Return: 
        t-ratio - ratio of areas (convex hull vs PCH)
     
    """
    ch_area  = ConvexHull(xp).volume
    pch_area  = ConvexHull(aa.T).volume
    return ch_area/pch_area

def get_relative_variation(aa_list):
    """noise / signal ratio (distance unit - not squared)
    noise: mean distance variation
    signal: mean pairwise distance between archetypes
    """
    # average across trials (for each noc, ndim)
    aa_avg = np.mean(aa_list, axis=0).T
    aa_std = np.std(aa_list, axis=0).T

    # signal - mean (across noc) distance (across ndim) to the nearest archetype
    signal = np.mean(np.sort(pairwise_distances(aa_avg), axis=0)[1])
    # noise - mean (across noc) distance diff (across ndim) around an archetype
    noise  = np.mean(np.sqrt(np.sum(np.power(aa_std,2), axis=1)))
    
    rv = noise/signal
    return rv 

def mean_archetype_dist(aa_samp, aa_global):
    """Mean distance between matched archetypes (after nearest-neighbor matching)."""
    D = pairwise_distances(aa_samp.T, aa_global.T)
    return np.mean(D.min(axis=1))

def plot_archetype(ax, aa, fmt='--o', color='k', **kwargs):
    """
    """
    ax.plot(aa[0].tolist()+[aa[0,0]], aa[1].tolist()+[aa[1,0]], fmt, color=color, **kwargs)
    