# Shared functions for archetype number selection scripts.

import numpy as np

from scomics.utils import get_relative_variation


def select_hvg(x_raw, depths_raw, n_top_genes):
    """Select top highly variable genes via normalize→log1p→variance ranking."""
    median_depth = np.median(depths_raw)
    x_norm = x_raw / depths_raw[:, None] * median_depth
    gene_var = np.log1p(x_norm).var(axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_top_genes]
    mask = np.zeros(x_raw.shape[1], dtype=bool)
    mask[top_idx] = True
    return mask


def run_noc_sweep(sca, noc_grid, ndim, nrepeats, groups):
    """Run PCHA + bootstrap ARV + per-group ARV for each NOC.

    Returns parallel lists: ev_grid, av_grid, av_rep_grid, xp_grid, aa_grid, aa_reps_grid.
    aa_reps_grid[i] is a list of (group_id, aa) pairs for NOC i.
    """
    group_ids = np.unique(groups)
    ev_grid, av_grid, av_rep_grid = [], [], []
    xp_grid, aa_grid, aa_reps_grid = [], [], []

    for noc in noc_grid:
        xp, aa, ev = sca.proj_and_pcha(ndim, noc)

        aa_boots = sca.bootstrap_proj_pcha(ndim, noc, nrepeats=nrepeats)
        av = get_relative_variation(aa_boots)

        aa_reps = []
        for g in group_ids:
            gmask = groups == g
            if gmask.sum() < max(noc * 5, 20):
                continue
            try:
                _, aa_g, _ = sca.pcha_on_subset(gmask, noc)
            except Exception:
                continue
            aa_reps.append((g, aa_g))
        av_rep = get_relative_variation([a for _, a in aa_reps]) if len(aa_reps) >= 2 else float('nan')

        print(f"  NOC={noc}  EV={ev:.4f}  ARV={av:.4f}  ARV_rep={av_rep:.4f}"
              f"  effEV={ev*(1-av):.4f}  effEV_rep={ev*(1-av_rep):.4f}")

        ev_grid.append(ev)
        av_grid.append(av)
        av_rep_grid.append(av_rep)
        xp_grid.append(xp)
        aa_grid.append(aa)
        aa_reps_grid.append(aa_reps)

    return (np.array(ev_grid), np.array(av_grid), np.array(av_rep_grid),
            xp_grid, aa_grid, aa_reps_grid)
