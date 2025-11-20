"""SingleCellArchetype Class
"""

import numpy as np
import pandas as pd
from scipy import stats

from .utils import *

class SCA():
    """
    """
    def __init__(self, norm_mat, types):
        """
        Arguments: 
            norm_mat - cell by gene feature matrix (depth normalized)
            types  - cell type labels per cell
        
        Initiate the SingleCellArchetype object

        """
        # input
        self.xn = norm_mat
        self.types = types
        
        # cell type label
        types_idx, types_lbl = pd.factorize(types, sort=True)
        
        self.types_idx = types_idx
        self.types_lbl = types_lbl 
        
        # feature matrix (init as the data - rather than shuffled)
        self.xf = self.xn 
        return 
        
    def setup_feature_matrix(self, method='data'):
        """
        """
        if method == 'data': 
            self.xf = self.xn
            print('use data')
            return  
        
        elif method == 'gshuff':
            # shuffle gene expression globally across all cells
            self.xf = shuffle_rows_per_col(self.xn)
            print('use shuffled data')
            return
            
        elif method == 'tshuff':
            # shuff each gene across cells independently - internally for each type A,B,C
            xn = self.xn
            xn_tshuff = xn.copy()
            
            types_lbl = self.types_lbl
            types_idx = self.types_idx
            for i in range(len(types_lbl)):
                xn_tshuff[types_idx==i] = shuffle_rows_per_col(xn[types_idx==i])
            self.xf = xn_tshuff
            print('use per-type shuffled data')
            return
        else:
            raise ValueError('choose from (data, gshuff, tshuff)')
    
    def proj_and_pcha(self, ndim, noc, **kwargs):
        """
        """
        xp = proj(self.xf, ndim)
        aa, varexpl = pcha(xp.T, noc=noc, **kwargs)
        
        self.xp = xp
        self.aa = aa
        self.varexpl = varexpl
        return (xp, aa, varexpl)
        
    def bootstrap_proj_pcha(self, ndim, noc, nrepeats=10, which='cell',
                           is_bootstrap=True,
                           downsamp_p=None,
                           preserve_embedding_sign=True, 
                           seed=None,
                           **kwargs): 
        """bootstrap or downsample (with a specified p)
        """
        if seed is not None:
            np.random.seed(seed)

        xp0 = proj(self.xf, ndim)
        
        aa_dsamps = []
        for i in range(nrepeats):
            xn_dsamp, cond_dsamp = bootstrap_or_downsamp(
                                    self.xf, 
                                    which=which, 
                                    is_bootstrap=is_bootstrap,
                                    downsamp_p=downsamp_p, 
                                    return_cond=True)

            xp_dsamp = proj(xn_dsamp, ndim)
            # match sign
            if preserve_embedding_sign:
                for i in range(ndim):
                    r, _ = stats.pearsonr(xp0[cond_dsamp,i], xp_dsamp[:,i])
                    sign = 2*int(r>0)-1
                    xp_dsamp[:,i] = sign*xp_dsamp[:,i]
            
            aa_dsamp, varexpl_dsamp = pcha(xp_dsamp.T, noc=noc, **kwargs)
            aa_dsamps.append(aa_dsamp)
            
        return aa_dsamps
    
    def t_ratio_test(self, ndim, noc, nrepeats=10, **kwargs): 
        """
        """
        
        self.setup_feature_matrix(method='data')
        xp, aa, varexpl = self.proj_and_pcha(ndim, noc)
        t_ratio = get_t_ratio(xp, aa)
        
        t_ratio_shuff_list = []
        for i in range(nrepeats):
            try: 
                self.setup_feature_matrix(method='gshuff')
                xp_shuff, aa_shuff, varexpl_shuff = self.proj_and_pcha(ndim, noc)
                t_ratio_shuff = get_t_ratio(xp_shuff, aa_shuff)
                t_ratio_shuff_list.append(t_ratio_shuff)
            except:
                print(f"skip repeat {i} - non convergence")
                nrepeats -= 1
        t_ratio_shuff_list = np.array(t_ratio_shuff_list)
            
        pvalue = (np.sum(t_ratio > t_ratio_shuff_list)+1)/nrepeats
        
        return t_ratio, t_ratio_shuff_list, pvalue
        