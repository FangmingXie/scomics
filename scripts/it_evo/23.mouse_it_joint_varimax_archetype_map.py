"""All mouse IT cells in one varimax space, colored by archetype — the mouse depth order.

Script 22's pipeline applied to Cheng22 mouse, so the two species' depth orders are derived
the same way. Script 15 already answers the mouse question on raw PC1-PC2; this does not
replace it. It exists because 15 and 22 do not use the same procedure, which makes their
outputs awkward to compare in a cross-species project, and because 15's basis choice was
never tested against the criteria 22 applies — it was justified by a variance partition
showing PC1/PC2 cell-type dominated, which is necessary but not sufficient (a component can
be uncontaminated and still carry no between-subclass contrast; see `informativeness`).

The two questions this answers that 15 cannot:
  1. Does the gate + subspace-PCA pipeline REPRODUCE 15's arc on mouse? On human the naive
     PCs were unusable and the pipeline was forced; on mouse PC1/PC2 look clean already, so
     mouse is the control where the extra machinery should change little. If it does change
     the order, 15's arc was resting on the basis rather than on the data.
  2. How much of mouse's cell-type structure sits in two axes compared with human's? The
     basis-comparison table is computed identically here and in 22, so the numbers are
     directly comparable — human needed varimax plus a second PCA to get a readable 2-D
     view, and whether mouse needs it is a fact about the species pair, not the method.

The new order is compared against 15's curated one at the end, both linearly and cyclically.

Differences from the human script 22, all forced by the mouse side:
  * One pooled IT h5ad instead of four per-subclass files, so the pooled `var` assertion has
    no analogue; instead the `Subclass` values are asserted to be exactly the four IT
    subclasses (15's check). The pooled file's cell order does not group subclasses, so the
    per-subclass assignment blocks are REINDEXED to the embedding order rather than asserted
    equal to it, as 22 can do.
  * `.X` is log1p(CP10k) but `.raw` holds counts, so the matrix is renormalized to
    log2(CP10k+1) from `.raw` (G2) — the opposite of the human side, where `.X` is used
    directly. This is 15's normalization, kept so the embedding stays comparable to it.
  * `var_names` are symbols already; there is no `feature_name` indirection (G5).
  * Three factors, not four: cell_type / sample / technical. Mouse has no donor-vs-source
    split; `Sample` is the batch analogue of human `donor_id`.
  * The technical block reads `n_counts`, `n_genes` and `percent_mito` straight from obs —
    all three are precomputed here, unlike human where the mitochondrial fraction had to be
    summed from `.raw`. log10 depth and `n_genes` correlate 0.95 (human 0.835), so grouping
    them into one factor matters more here, not less.
  * At 11,061 x 16,572 the pooled matrix is 733 MB dense, so 22's sparse-moment machinery
    (float64 `np.bincount` over CSR indices, to avoid a 10.9 GB densify) is unnecessary and
    the matrix is densified as 15 does.

Caveats:
  * Pooled HVGs are dominated by between-subclass genes. That is the point here, but it
    makes this a new space, not comparable to the per-subclass varimax records, and it must
    not be used to re-derive archetypes.
  * Cell counts are unbalanced (L4 4732 / L2/3 4044 / L6IT 1436 / L5IT 849), though far less
    than human's 50.6/32.7/11.3/5.4 %.
  * 05's top-cell counts are NOT uniform — 300 for L23/L4 but 100 for L5IT/L6IT — so the
    centroids are not all averages of equally many cells, unlike human's uniform 300. The
    all-cell (size-weighted) centroid order is reported as the control on this.
  * This script's tables keep 05's A/B/C letters. The primed labels it proposes live only in
    OUT_ARC_ORDER's `new_label`, and 12/14 continue to read 15's file, not this one.
  * PCHA's convex weights S would be the canonical archetype assignment, but 05 does not
    save them. Both proxies are emitted — nearest vertex in that subclass's own PCHA space,
    and argmax of 05's scores — plus a top-cells-only label.

THE ARC ORDER IS PRODUCED IN TWO RUNS — see ARC_ORDER_CURATED. Run 1 writes the raw angular
sort with `curated=False` in every row; that file is a PROPOSAL and any consumer must refuse
it. Only after the diagnostics and figures have been read does ARC_ORDER_CURATED get filled
in and the file become a depth record.

Reads:
  links/it_evo/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_{xp,aa}.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_scores.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_top_cells.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv   (for the comparison only)
Outputs:
  local_data/res/it_evo/23.mouse_IT_joint_pca_{coords,loadings}.tsv
  local_data/res/it_evo/23.mouse_IT_joint_varimax_{coords,loadings}.tsv
  local_data/res/it_evo/23.mouse_IT_joint_spc_coords.tsv
  local_data/res/it_evo/23.mouse_IT_joint_spc_loadings_{on_vx,gene}.tsv
  local_data/res/it_evo/23.mouse_IT_joint_{pc,vx,spc}_variance_partition.tsv
  local_data/res/it_evo/23.mouse_IT_joint_component_informativeness.tsv
  local_data/res/it_evo/23.mouse_IT_joint_plane_scan.tsv
  local_data/res/it_evo/23.mouse_IT_joint_basis_comparison.tsv
  local_data/res/it_evo/23.mouse_IT_joint_archetype_{assignment,centroids}.tsv
  local_data/res/it_evo/23.mouse_IT_joint_archetype_arc_order.tsv
  local_data/fig/it_evo/23.mouse_IT_joint_{pc,vx,spc}_{archetype,subclass}.pdf
"""

import os
import sys
import gc
import itertools
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_archetype_scatter_pdf

# --- file paths ---
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
INPUT_MOUSE   = os.path.join(PROJECT_ROOT, 'links', 'it_evo',
                             'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
IN_ARC_15     = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
OUT_PC_COORDS = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_pca_coords.tsv')
OUT_PC_LOAD   = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_pca_loadings.tsv')
OUT_VX_COORDS = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_varimax_coords.tsv')
OUT_VX_LOAD   = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_varimax_loadings.tsv')
OUT_SPC_COORDS = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_spc_coords.tsv')
OUT_SPC_LOAD_VX = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_spc_loadings_on_vx.tsv')
OUT_SPC_LOAD_GN = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_spc_loadings_gene.tsv')
OUT_PC_VAR    = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_pc_variance_partition.tsv')
OUT_VX_VAR    = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_vx_variance_partition.tsv')
OUT_SPC_VAR   = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_spc_variance_partition.tsv')
OUT_INFORM    = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_component_informativeness.tsv')
OUT_SCAN      = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_plane_scan.tsv')
OUT_BASIS     = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_basis_comparison.tsv')
OUT_ASSIGN    = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_archetype_assignment.tsv')
OUT_CENTROIDS = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_archetype_centroids.tsv')
OUT_ARC_ORDER = os.path.join(OUT_RES_DIR, '23.mouse_IT_joint_archetype_arc_order.tsv')
OUT_PDF = {b: {c: os.path.join(OUT_FIG_DIR, f'23.mouse_IT_joint_{b}_{c}.pdf')
               for c in ('archetype', 'subclass')}
           for b in ('pc', 'vx', 'spc')}

# `noc` must match script 05's noc for the same token.
SUBCLASSES = [
    {'token': 'L23',  'mouse_subclass': 'L2/3', 'noc': 3},
    {'token': 'L4',   'mouse_subclass': 'L4',   'noc': 3},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT', 'noc': 2},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT', 'noc': 3},
]

# --- parameters (column names from 15 / it/19) ---
SUBCLASS_COL      = 'Subclass'
CLUSTER_COL       = 'Type'
SAMPLE_COL        = 'Sample'
LIBSIZE_COL       = 'n_counts'
NGENE_COL         = 'n_genes'
MITO_COL          = 'percent_mito'
N_HVG             = 2000
N_PCS             = 10
PCA_RANDOM_STATE  = 0
ALPHABET          = ['A', 'B', 'C', 'D', 'E', 'F']
N_CELLS_EXPECTED  = {'L23': 4044, 'L4': 4732, 'L5IT': 849, 'L6IT': 1436}
N_CLUSTERS_POOLED = 9
N_SAMPLES         = 4
ARC_BINS          = 36          # 10-degree bins around the arc, for the connectedness check
INTERIOR_FRAC     = 0.5         # centroid radius below this x median -> its angle is unreliable
KNN_K             = 25
KNN_N             = 20000       # more than the 11,061 cells, so every cell is used
KNN_SEED          = 0
DIR_FIRST, DIR_LAST = 'L2/3', 'L6IT'

# ===========================================================================
# Gate A on the joint embedding, then the plane the geometry is read in — identical to 22.
# ARC_SPC is the top two subspace PCs by variance: no selection, and anatomical depth order
# plays no part in it, so the subclass sequence the arc comes out in stays independent
# evidence. PANEL_OVERRIDE fixes which components each basis is PLOTTED on (default: that
# basis's three most subclass-informative gated components); SPC uses its leading three for
# legibility, which cannot affect the comparison because SPC1-SPC2 is panel 1 either way.
# ===========================================================================
ARC_SPC        = ('SPC1', 'SPC2')
PANEL_OVERRIDE = {'spc': ('SPC1', 'SPC2', 'SPC3')}
# ===========================================================================

# ===========================================================================
# RUN 1 writes the raw angular sort with curated=False; RUN 2 records a curated order with
# the angular rank kept alongside. 15's curated mouse order is NOT pasted in here — that
# would presuppose the answer this script exists to test. It is compared against at the end.
# ===========================================================================
ARC_ORDER_CURATED = None
# ARC_ORDER_CURATED = [
#     'L2/3 ?', 'L2/3 ?', 'L2/3 ?',
#     'L4 ?',   'L4 ?',   'L4 ?',
#     'L5IT ?', 'L5IT ?',
#     'L6IT ?', 'L6IT ?', 'L6IT ?',
# ]
# ===========================================================================

UNCURATED_BANNER = (
    '\n' + '*' * 78 +
    '\n*** ARC_ORDER_CURATED IS None — THIS IS RUN 1 (UNCURATED).\n'
    '*** The arc order written below is the raw angular sort. It is a PROPOSAL, not the\n'
    '*** depth record. Inspect the PDFs and the geometry diagnostics, then set\n'
    '*** ARC_ORDER_CURATED and rerun. Every row of the arc-order TSV is marked\n'
    '*** curated=False; no consumer may treat a curated=False file as the depth order.\n'
    + '*' * 78 + '\n')

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

if ARC_ORDER_CURATED is None:
    print(UNCURATED_BANNER)

subclass_of = lambda n: n.rsplit(' ', 1)[0]


def varimax(L, gamma=1.0, max_iter=1000, tol=1e-6):
    """Kaiser varimax rotation of loading matrix L (n_vars × n_factors). Copied from 02:86."""
    n, p = L.shape
    R = np.eye(p)
    for _ in range(max_iter):
        R_old = R.copy()
        for i in range(p - 1):
            for j in range(i + 1, p):
                Lr = L @ R
                u = Lr[:, i] ** 2 - Lr[:, j] ** 2
                v = 2 * Lr[:, i] * Lr[:, j]
                A, B = u.sum(), v.sum()
                C, D = (u ** 2 - v ** 2).sum(), 2 * (u * v).sum()
                theta = 0.25 * np.arctan2(D - gamma * 2 * A * B / n,
                                          C - gamma * (A ** 2 - B ** 2) / n)
                c, s = np.cos(theta), np.sin(theta)
                Rij = np.eye(p)
                Rij[i, i] = Rij[j, j] = c
                Rij[i, j], Rij[j, i] = -s, s
                R = R @ Rij
        if np.max(np.abs(R - R_old)) < tol:
            break
    return R


def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def build_factors(cluster, sample, tech):
    """Three factors for mouse: cell_type / sample / technical.

    Strings, not Categoricals, for the dummies (G6: an unused level would become an all-zero
    column). `tech` is a (n_cells, 3) block — log10 library size, genes detected, mito
    fraction — entered as ONE factor because the three are collinear (r = 0.95 for the first
    two on this dataset) and partialling them separately would attribute their shared
    variance to neither.
    """
    for name, arr in (('cell_type', cluster), ('sample', sample)):
        if getattr(arr, 'dtype', None) == 'category':
            raise ValueError(f'{name} was passed as a Categorical — pool it as str (G6)')
    zero_var = [TECH_NAMES[j] for j in range(tech.shape[1]) if tech[:, j].std() == 0]
    if zero_var:
        raise ValueError(f'technical covariate(s) {zero_var} are constant — z-scoring them '
                         f'would divide by zero and the block would silently carry a NaN '
                         f'column into every regression')
    factors = {
        'cell_type': pd.get_dummies(cluster).values.astype(float),
        'sample':    pd.get_dummies(sample).values.astype(float),
        'technical': np.apply_along_axis(zscore, 0, tech),
    }
    expected = {'cell_type': N_CLUSTERS_POOLED, 'sample': N_SAMPLES}
    for name, n_exp in expected.items():
        col_sums = factors[name].sum(axis=0)
        if (col_sums == 0).any():
            raise ValueError(f'{name} design matrix has {(col_sums == 0).sum()} all-zero '
                             f'dummy column(s) — stale categorical levels survived (G6)')
        if factors[name].shape[1] != n_exp:
            raise ValueError(f'pooled {name} has {factors[name].shape[1]} levels, expected '
                             f'{n_exp} — this is not the IT subset the script was written '
                             f'against')
    return factors


def variance_partition(scores_arr, comp_cols, factors, tech, tech_names, label):
    """Partial R² per component for each factor (02's dict signature).

    `tech` / `tech_names` are only for the printed per-covariate marginal R², which keeps the
    grouped `technical` block interpretable — they take no part in the partition itself.
    """
    X_full = np.hstack(list(factors.values()))

    print(f'\nVariance partitioning ({label}):')
    rows = []
    for i, col in enumerate(comp_cols):
        y = scores_arr[:, i]
        r2_full = r2(X_full, y)
        partial = {}
        for name in factors:
            X_others = np.hstack([v for k, v in factors.items() if k != name])
            partial[name] = max(r2_full - r2(X_others, y), 0.0)
        residual = max(1.0 - r2_full, 0.0)
        marg = {f'r2_{n}': r2(tech[:, [j]], y) for j, n in enumerate(tech_names)}
        rows.append({'component': col, **partial, 'residual': residual, **marg})
        print(f'  {col:5s} R²_full={r2_full:.3f}  '
              + '  '.join(f'{k}={v:.3f}' for k, v in partial.items())
              + f'  resid={residual:.3f}   [marginal '
              + ' '.join(f'{n}={marg[f"r2_{n}"]:.2f}' for n in tech_names) + ']')
    return pd.DataFrame(rows).set_index('component')


def subclass_runs(arc):
    """Subclass sequence along `arc`, collapsing only CONSECUTIVE repeats.

    `drop_duplicates` would collapse non-consecutive ones too, so an arc that leaves a
    subclass and comes back to it would be reported as tidy contiguous blocks.
    """
    return [k for k, _ in itertools.groupby(subclass_of(n) for n in arc)]


def cyclic_offset(order_a, order_b):
    """Is `order_b` the same cycle as `order_a`? Returns (rotation, reversed) or None.

    The centroids can form a closed-ish loop, in which case an ordering is only defined up to
    where the circle was cut and which way it was traversed; a plain inverted-pair count
    charges the full cost of a rotation.
    """
    n = len(order_a)
    for rev in (False, True):
        b = order_b[::-1] if rev else list(order_b)
        for k in range(n):
            if b[k:] + b[:k] == list(order_a):
                return k, rev
    return None


def inverted_pairs(order_a, order_b):
    """Pairs ordered one way in `order_a` and the other way in `order_b`."""
    rank_b = {n: i for i, n in enumerate(order_b)}
    return [(a, b) for a, b in itertools.combinations(order_a, 2) if rank_b[a] > rank_b[b]]


def arc_from_centroids(P):
    """Order the centroids around the arc in the P (2-column) plane.

    The circle is cut at the LARGEST ANGULAR GAP rather than at the -180 deg wraparound, and
    the traversal is oriented so DIR_FIRST precedes DIR_LAST — a component's sign is
    arbitrary, so this fixes a READING DIRECTION only, not the order up to reversal.
    """
    centre = P.mean()
    angle  = pd.Series(np.degrees(np.arctan2(*(P - centre).values.T[::-1])), index=P.index)

    ang_sorted = angle.sort_values()
    names      = list(ang_sorted.index)
    gaps       = np.diff(np.append(ang_sorted.values, ang_sorted.values[0] + 360.0))
    cut        = int(np.argmax(gaps))
    arc        = names[cut + 1:] + names[:cut + 1]

    mean_rank = lambda o, sub: np.mean([i for i, n in enumerate(o) if subclass_of(n) == sub])
    if mean_rank(arc, DIR_FIRST) > mean_rank(arc, DIR_LAST):
        arc = arc[::-1]
    return arc, angle, centre, float(gaps[cut])


# ---------------------------------------------------------------------------
# 1. All IT cells, one pooled file
# ---------------------------------------------------------------------------
print(f'Loading {INPUT_MOUSE}...')
adata = ad.read_h5ad(INPUT_MOUSE)
print(f'  {adata.n_obs} cells x {adata.n_vars} genes')

expected = {cfg['mouse_subclass'] for cfg in SUBCLASSES}
present  = set(pd.unique(adata.obs[SUBCLASS_COL].astype(str)))
if present != expected:
    raise ValueError(f'h5ad {SUBCLASS_COL} values {sorted(present)} != the four IT subclasses '
                     f'{sorted(expected)} — the joint embedding assumes the file is exactly '
                     f'the cells the per-subclass records were fit on')
counts = adata.obs[SUBCLASS_COL].value_counts().to_dict()
print('  cells per subclass:', counts)
for cfg in SUBCLASSES:
    n_exp = N_CELLS_EXPECTED[cfg['token']]
    if counts[cfg['mouse_subclass']] != n_exp:
        raise ValueError(f'{cfg["token"]}: {counts[cfg["mouse_subclass"]]} cells but {n_exp} '
                         f'expected — this is not the subset 05 was fit on')

barcodes   = adata.obs_names.values
subclasses = adata.obs[SUBCLASS_COL].astype(str).values
clusters   = adata.obs[CLUSTER_COL].astype(str).values
samples    = adata.obs[SAMPLE_COL].astype(str).values
TECH_NAMES = ['log10_libsize', 'n_genes', 'mito_frac']
tech = np.column_stack([np.log10(adata.obs[LIBSIZE_COL].values.astype(float)),
                        adata.obs[NGENE_COL].values.astype(float),
                        adata.obs[MITO_COL].values.astype(float)])
n_pooled = adata.n_obs
print('  technical covariates (median): '
      + ', '.join(f'{n}={np.median(tech[:, j]):.4f}' for j, n in enumerate(TECH_NAMES))
      + f'   r(log10_libsize, n_genes)={np.corrcoef(tech[:, 0], tech[:, 1])[0, 1]:.3f}')

# --- 2. Normalize: raw counts -> log2(CP10k + 1)  (G2, 15:197-204) ---
if adata.raw is None or not np.array_equal(adata.raw.var_names.values, adata.var_names.values):
    raise ValueError('.raw is missing or its var differs from .X — G2 renormalization assumes '
                     'the same genes in the same order')
print('Normalizing (log2(CP10k+1) from .raw counts)...')
X_raw  = adata.raw.X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm = np.log2(X_raw / depths * 1e4 + 1)
del X_raw
gc.collect()

# --- 3. HVG (15:206-211). 11,061 x 16,572 float32 is 733 MB, so no sparse-moment pass ---
print(f'Selecting top {N_HVG} HVGs across all IT cells...')
gene_var = X_norm.var(axis=0)
order    = np.argsort(gene_var)[::-1]
gap      = gene_var[order[N_HVG - 1]] - gene_var[order[N_HVG]]
print(f'  variance at the cutoff: rank {N_HVG - 1}={gene_var[order[N_HVG - 1]]:.6f}, '
      f'rank {N_HVG}={gene_var[order[N_HVG]]:.6f}, gap={gap:.3e}')
if gap <= 0:
    raise ValueError(f'exact variance tie at the {N_HVG}-gene cutoff — the HVG set would be '
                     f'argsort-order-dependent, not a property of the data')
hvg_idx   = order[:N_HVG]
hvg_names = adata.var_names.values[hvg_idx]
X_hvg     = X_norm[:, hvg_idx]
del X_norm
gc.collect()

# --- 4. Scale + PCA + varimax ---
print('Scaling and PCA...')
X_scaled = StandardScaler().fit_transform(X_hvg)
del X_hvg
gc.collect()
pca     = PCA(n_components=N_PCS, random_state=PCA_RANDOM_STATE)
scores  = pca.fit_transform(X_scaled)
L       = pca.components_.T
pc_cols = [f'PC{i + 1}' for i in range(N_PCS)]
del X_scaled
gc.collect()
print('  explained variance ratio: '
      + ', '.join(f'{c}={v:.3f}' for c, v in zip(pc_cols, pca.explained_variance_ratio_)))

pd.DataFrame(scores, index=barcodes, columns=pc_cols).to_csv(OUT_PC_COORDS, sep='\t')
pd.DataFrame(L, index=hvg_names, columns=pc_cols).to_csv(OUT_PC_LOAD, sep='\t')
print(f'Saved {OUT_PC_COORDS} and {OUT_PC_LOAD}')

print('Running varimax...')
R           = varimax(L)
vx_scores   = scores @ R
vx_loadings = L @ R
vx_order    = np.argsort(vx_scores.var(axis=0))[::-1]      # 02: descending score variance
vx_scores   = vx_scores[:, vx_order]
vx_loadings = vx_loadings[:, vx_order]
vx_cols     = [f'VX{i + 1}' for i in range(N_PCS)]
print('  VX score-variance share: '
      + ', '.join(f'{c}={v:.3f}' for c, v in
                  zip(vx_cols, vx_scores.var(axis=0) / vx_scores.var(axis=0).sum())))

pd.DataFrame(vx_scores, index=barcodes, columns=vx_cols).to_csv(OUT_VX_COORDS, sep='\t')
pd.DataFrame(vx_loadings, index=hvg_names, columns=vx_cols).to_csv(OUT_VX_LOAD, sep='\t')
print(f'Saved {OUT_VX_COORDS} and {OUT_VX_LOAD}')

# ---------------------------------------------------------------------------
# 5. Archetype assignment per cell, from the per-subclass PCHA records of script 05
# ---------------------------------------------------------------------------
print('\nAssigning cells to archetypes...')
blocks = []
for cfg in SUBCLASSES:
    token, subclass, noc = cfg['token'], cfg['mouse_subclass'], cfg['noc']
    xp_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv'),
                            sep='\t', index_col=0)
    aa_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv'),
                            sep='\t', index_col=0)
    scores_df = pd.read_csv(os.path.join(OUT_RES_DIR,
                                         f'05.mouse_{token}_archetype_scores.tsv'),
                            sep='\t', index_col=0)
    top_df    = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_top_cells.tsv'),
                            sep='\t')

    if len(aa_df) != noc:
        raise ValueError(f'{token}: pcha_aa has {len(aa_df)} archetypes but noc={noc}')
    if list(aa_df.columns) != list(xp_df.columns):
        raise ValueError(f'{token}: pcha_aa columns {list(aa_df.columns)} != pcha_xp columns '
                         f'{list(xp_df.columns)} — the vertices are not in the cells\' space')
    sub_cells = set(adata.obs_names[adata.obs[SUBCLASS_COL].astype(str) == subclass])
    if sub_cells != set(xp_df.index):
        raise ValueError(f'{token}: the {len(sub_cells)} {subclass} cells in the h5ad are not '
                         f'the {len(xp_df)} cells of 05.mouse_{token}_pcha_xp.tsv')
    if not scores_df.index.equals(xp_df.index):
        raise ValueError(f'{token}: 05 scores and 05 pcha_xp disagree on cell order')
    if not set(top_df['cell']).issubset(set(xp_df.index)):
        raise ValueError(f'{token}: 05 top_cells names cells absent from 05 pcha_xp')

    names = [f'{subclass} {ALPHABET[k]}' for k in range(noc)]

    d = np.stack([np.linalg.norm(xp_df.values - aa_df.values[k], axis=1) for k in range(noc)],
                 axis=1)
    nearest = np.array(names)[d.argmin(axis=1)]

    score_cols = [f'score_{ALPHABET[k]}' for k in range(noc)]
    by_score = np.array(names)[scores_df[score_cols].values.argmax(axis=1)]

    top_label = pd.Series('other', index=xp_df.index)
    for k in range(noc):
        cells = top_df[top_df['archetype'] == f'archetype_{k + 1}']['cell'].values
        top_label.loc[cells] = names[k]

    blocks.append(pd.DataFrame({'mouse_subclass': subclass,
                                'archetype': nearest,
                                'archetype_by_score': by_score,
                                'archetype_top_cells': top_label.values},
                               index=xp_df.index))
    agree = (nearest == by_score).mean()
    print(f'  {token}: {len(xp_df)} cells, nearest-vertex vs score-argmax agree {agree:.1%}')
    print(f'    counts: {pd.Series(nearest).value_counts().to_dict()}')

assign = pd.concat(blocks)
if len(assign) != n_pooled or not assign.index.is_unique:
    raise ValueError(f'assignment covers {len(assign)} rows ({assign.index.nunique()} unique) '
                     f'but the pool has {n_pooled} cells — every cell must be assigned once')
# The pooled h5ad does not group subclasses, so unlike 22 this is a reindex, not an assertion.
assign = assign.loc[barcodes]
assign[CLUSTER_COL] = clusters
assign[SAMPLE_COL]  = samples
assign.to_csv(OUT_ASSIGN, sep='\t')
print(f'Saved {OUT_ASSIGN}')

ARCH_ORDER = [f'{cfg["mouse_subclass"]} {ALPHABET[k]}'
              for cfg in SUBCLASSES for k in range(cfg['noc'])]

# ---------------------------------------------------------------------------
# 6. What do the components encode, and which are cell-type driven
# ---------------------------------------------------------------------------
factors = build_factors(clusters, samples, tech)
pc_var  = variance_partition(scores, pc_cols, factors, tech, TECH_NAMES,
                             'joint mouse IT PCs — pre-varimax')
pc_var.to_csv(OUT_PC_VAR, sep='\t')
vx_var  = variance_partition(vx_scores, vx_cols, factors, tech, TECH_NAMES,
                             'joint mouse IT VX — post-varimax')
vx_var.to_csv(OUT_VX_VAR, sep='\t')
print(f'Saved {OUT_PC_VAR} and {OUT_VX_VAR}')

TECH_FACTORS = ('sample', 'technical')


def gate_of(var_df, cols, label):
    """Components whose cell_type partial R² beats every technical factor's."""
    keep = [c for c in cols
            if all(var_df.loc[c, 'cell_type'] > var_df.loc[c, t] for t in TECH_FACTORS)]
    print(f'\n{label}:')
    for c in cols:
        worst = max(TECH_FACTORS, key=lambda t: var_df.loc[c, t])
        print(f'  {"KEEP  " if c in keep else "reject"} {c:5s} '
              f'cell_type={var_df.loc[c, "cell_type"]:.3f}  '
              f'largest technical: {worst}={var_df.loc[c, worst]:.3f}')
    return keep


gate_pc = gate_of(pc_var, pc_cols,
                  'Naive PCs the partition calls cell-type driven (reported for comparison; '
                  'nothing selects on it)')
gate_a  = gate_of(vx_var, vx_cols,
                  'Joint-embedding Gate A — VX components the partition calls cell-type driven')
if len(gate_a) < 3:
    raise ValueError(f'Gate A kept only {len(gate_a)} components ({gate_a}); the subspace PCA '
                     f'and its three-panel figures need at least 3')

vx_df_all = pd.DataFrame(vx_scores, index=barcodes, columns=vx_cols)

# ---------------------------------------------------------------------------
# 7. PCA inside the gated subspace — a rotation of it, putting the cell-type structure back
#    into leading axes so a 2-D geometry can be read
# ---------------------------------------------------------------------------
gate_idx   = [vx_cols.index(c) for c in gate_a]
spc_pca    = PCA(n_components=len(gate_a), random_state=PCA_RANDOM_STATE)
spc_scores = spc_pca.fit_transform(vx_scores[:, gate_idx])
spc_cols   = [f'SPC{i + 1}' for i in range(len(gate_a))]
spc_df_all = pd.DataFrame(spc_scores, index=barcodes, columns=spc_cols)
print(f'\nPCA within the {len(gate_a)}-component Gate-A subspace -> {spc_cols}')
print('  explained variance ratio: '
      + ', '.join(f'{c}={v:.3f}' for c, v in zip(spc_cols, spc_pca.explained_variance_ratio_))
      + f'   (top two {spc_pca.explained_variance_ratio_[:2].sum():.3f})')

spc_df_all.to_csv(OUT_SPC_COORDS, sep='\t')
pd.DataFrame(spc_pca.components_.T, index=gate_a, columns=spc_cols).to_csv(OUT_SPC_LOAD_VX,
                                                                          sep='\t')
pd.DataFrame(vx_loadings[:, gate_idx] @ spc_pca.components_.T,
             index=hvg_names, columns=spc_cols).to_csv(OUT_SPC_LOAD_GN, sep='\t')
print(f'Saved {OUT_SPC_COORDS}, {OUT_SPC_LOAD_VX} and {OUT_SPC_LOAD_GN}')

spc_var = variance_partition(spc_scores, spc_cols, factors, tech, TECH_NAMES,
                             'joint mouse IT SPC — PCs of the gated subspace')
spc_var.to_csv(OUT_SPC_VAR, sep='\t')
print(f'Saved {OUT_SPC_VAR}')
leaked = [c for c in spc_cols
          if any(spc_var.loc[c, 'cell_type'] <= spc_var.loc[c, t] for t in TECH_FACTORS)]
if leaked:
    raise ValueError(f'{leaked} are not cell-type dominated, but every SPC is a combination '
                     f'of Gate-A components only — a technical factor cannot enter here '
                     f'unless the gate or the partition is inconsistent')

# ---------------------------------------------------------------------------
# 8. Which components actually carry cell identity, in all three bases
# ---------------------------------------------------------------------------
X_SUBCLASS  = pd.get_dummies(subclasses).values.astype(float)
X_ARCHETYPE = pd.get_dummies(assign['archetype'].values).values.astype(float)


def informativeness(scores_arr, cols, var_df, gate, basis):
    """How much of each component is cell identity, at the two levels that matter.

    `cell_type` in the partition is the 9 `Type` levels partialled against the technical
    factors, which answers "is this axis contaminated". It does NOT answer "does this axis
    carry the between-subclass contrast the arc is read from" — a component can be cleanly
    cell-type driven and still describe only within-subclass variation. `subclass_r2` and
    `archetype_r2` are plain marginal R² on the four subclasses and the 11 archetypes.
    """
    rows = [{'basis': basis, 'component': c,
             'cell_type_partial': round(var_df.loc[c, 'cell_type'], 3),
             'subclass_r2': round(r2(X_SUBCLASS, scores_arr[:, i]), 3),
             'archetype_r2': round(r2(X_ARCHETYPE, scores_arr[:, i]), 3),
             'sample': round(var_df.loc[c, 'sample'], 3),
             'technical': round(var_df.loc[c, 'technical'], 3),
             'gate_pass': c in gate}
            for i, c in enumerate(cols)]
    df = pd.DataFrame(rows).sort_values('subclass_r2', ascending=False)
    print(f'\n{basis.upper()} components ranked by between-subclass informativeness:')
    print(df.drop(columns=['basis']).to_string(index=False))
    return df


inform = pd.concat([informativeness(scores, pc_cols, pc_var, gate_pc, 'pc'),
                    informativeness(vx_scores, vx_cols, vx_var, gate_a, 'vx'),
                    informativeness(spc_scores, spc_cols, spc_var, spc_cols, 'spc')])
inform.to_csv(OUT_INFORM, sep='\t', index=False)
print(f'\nSaved {OUT_INFORM}')

# ---------------------------------------------------------------------------
# 9. Centroids, and the diagnostics that say whether a 2-D read is legitimate
# ---------------------------------------------------------------------------
centroids = pd.DataFrame(
    [spc_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
centroids.to_csv(OUT_CENTROIDS, sep='\t')
print(f'Saved {OUT_CENTROIDS}')

rng      = np.random.default_rng(KNN_SEED)
knn_idx  = rng.choice(n_pooled, min(KNN_N, n_pooled), replace=False)
knn_sub  = subclasses[knn_idx]
knn_null = float((pd.Series(knn_sub).value_counts(normalize=True) ** 2).sum())


def knn_purity(frame, cols):
    X = frame.iloc[knn_idx][list(cols)].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    _, nb = cKDTree(X).query(X, k=KNN_K + 1)
    return float((knn_sub[nb[:, 1:]] == knn_sub[:, None]).mean())


def plane_scan(frame, cent_frame, cols, label, ceiling):
    print(f'\n{label} — kNN(k={KNN_K}) subclass purity on {len(knn_idx)} cells; '
          f'chance={knn_null:.3f}, full {len(cols)}-component subspace={ceiling:.3f}:')
    rows = []
    for x, y in itertools.combinations(cols, 2):
        arc_xy, _, _, gap_xy = arc_from_centroids(cent_frame[[x, y]])
        runs_xy = subclass_runs(arc_xy)
        rows.append({'plane': f'{x}-{y}', 'x': x, 'y': y,
                     'knn_purity': round(knn_purity(frame, (x, y)), 3),
                     'subclass_seq': ' '.join(runs_xy),
                     'n_runs': len(runs_xy),
                     'contiguous': len(runs_xy) == len(SUBCLASSES),
                     'depth_order': runs_xy == [c['mouse_subclass'] for c in SUBCLASSES]})
    df = pd.DataFrame(rows).sort_values('knn_purity', ascending=False)
    print(f'  {int(df["contiguous"].sum())} of {len(df)} planes keep every subclass '
          f'contiguous; {int(df["depth_order"].sum())} also give the anatomical order')
    print(df.drop(columns=['x', 'y']).to_string(index=False))
    return df


pc_df_all    = pd.DataFrame(scores, index=barcodes, columns=pc_cols)
centroids_pc = pd.DataFrame(
    [pc_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
centroids_vx = pd.DataFrame(
    [vx_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
pc_ceiling  = knn_purity(pc_df_all, gate_pc)
vx_ceiling  = knn_purity(vx_df_all, gate_a)
spc_ceiling = knn_purity(spc_df_all, spc_cols)
scan_pc  = plane_scan(pc_df_all, centroids_pc, gate_pc,
                      f'Plane scan over the {len(gate_pc)} gate-passing naive PCs', pc_ceiling)
scan_vx  = plane_scan(vx_df_all, centroids_vx, gate_a,
                      f'Plane scan over the {len(gate_a)} Gate-A VX components', vx_ceiling)
scan_spc = plane_scan(spc_df_all, centroids, spc_cols,
                      f'Plane scan over the {len(spc_cols)} subspace PCs (SPC)', spc_ceiling)
scan_df = pd.concat([scan_pc.assign(basis='pc'), scan_vx.assign(basis='vx'),
                     scan_spc.assign(basis='spc')])
scan_df.to_csv(OUT_SCAN, sep='\t', index=False)
print(f'\nSaved {OUT_SCAN}')

basis_rows = []
for name, cols_b, ceil_b, scan_b in (('pc', gate_pc, pc_ceiling, scan_pc),
                                     ('vx', gate_a, vx_ceiling, scan_vx),
                                     ('spc', spc_cols, spc_ceiling, scan_spc)):
    best = scan_b.iloc[0]
    basis_rows.append({'basis': name, 'n_components': len(cols_b),
                       'subspace_purity': round(ceil_b, 3),
                       'best_plane': best['plane'],
                       'best_plane_purity': best['knn_purity'],
                       'purity_kept': round(best['knn_purity'] / ceil_b, 3),
                       'n_contiguous_planes': int(scan_b['contiguous'].sum()),
                       'n_depth_order_planes': int(scan_b['depth_order'].sum())})
basis_df = pd.DataFrame(basis_rows)
basis_df.to_csv(OUT_BASIS, sep='\t', index=False)
print(f'\nBasis comparison — chance purity {knn_null:.3f} '
      f'(directly comparable to 22.human_IT_joint_basis_comparison.tsv):')
print(basis_df.to_string(index=False))
print(f'Saved {OUT_BASIS}')

bad = [c for c in ARC_SPC if c not in spc_cols]
if bad:
    raise ValueError(f'ARC_SPC names {bad}, which are not among {spc_cols}')
print(f'\nGeometry is read on {ARC_SPC} — the top two subspace PCs by variance, '
      f'no selection applied.')
print(f'  purity in this plane {knn_purity(spc_df_all, ARC_SPC):.3f} vs {spc_ceiling:.3f} for '
      f'the full {len(spc_cols)}-component subspace '
      f'(best gated VX plane managed {scan_vx["knn_purity"].max():.3f}, '
      f'best naive-PC plane {scan_pc["knn_purity"].max():.3f})')

# ---------------------------------------------------------------------------
# 10. Figures — each basis on its own most subclass-informative gated components, except
#     where PANEL_OVERRIDE says otherwise. Plotting the first three of each basis instead
#     would put PC and VX on whatever axes the gate rejects and rig the comparison by panel
#     choice. No archetype overlay: the shared helper closes a polygon through the vertices,
#     meaningless across four separately-fit simplices.
# ---------------------------------------------------------------------------
print('\nGenerating figures...')
BASES = [('pc',  scores,     pc_cols,  gate_pc),
         ('vx',  vx_scores,  vx_cols,  gate_a),
         ('spc', spc_scores, spc_cols, spc_cols)]
for basis, coords, cols, gate in BASES:
    ranked = inform[(inform['basis'] == basis) & inform['gate_pass']].sort_values(
        'subclass_r2', ascending=False)
    if len(ranked) < 3:
        raise ValueError(f'{basis}: only {len(ranked)} gated components, need 3 for the panels')
    by_component = ranked.set_index('component')
    if basis in PANEL_OVERRIDE:
        top3 = list(PANEL_OVERRIDE[basis])
        bad  = [c for c in top3 if c not in by_component.index]
        if bad:
            raise ValueError(f'PANEL_OVERRIDE[{basis!r}] names {bad}, which are not gated '
                             f'components of that basis')
        note = ' (leading three, by variance)'
    else:
        top3 = list(ranked['component'])[:3]
        note = ''
    r2s  = {c: by_component.loc[c, 'subclass_r2'] for c in top3}
    idx  = [cols.index(c) for c in top3]
    pan  = [(idx[0], idx[1], top3[0], top3[1]),
            (idx[0], idx[2], top3[0], top3[2]),
            (idx[1], idx[2], top3[1], top3[2])]
    best = basis_df.set_index('basis').loc[basis]
    blurb = (f'{basis.upper()} basis — gated components{note} '
             f'(' + ', '.join(f'{c} subclass R²={r2s[c]:.2f}' for c in top3) + f'); best plane '
             f'{best["best_plane"]} keeps {best["purity_kept"]:.2f} of this basis\'s '
             f'{best["subspace_purity"]:.3f} subspace purity')
    print(f'  {basis}: panels on {top3}')
    save_archetype_scatter_pdf(
        coords, assign['archetype'].values, pan, aa=None,
        title=f'Cheng22 mouse IT joint embedding, by archetype ({n_pooled} cells, '
              f'05 letters)\n{blurb}',
        out_path=OUT_PDF[basis]['archetype'], label_order=ARCH_ORDER, cmap='tab20', s=2)
    save_archetype_scatter_pdf(
        coords, subclasses, pan, aa=None,
        title=f'Cheng22 mouse IT joint embedding, by subclass ({n_pooled} cells)\n{blurb}',
        out_path=OUT_PDF[basis]['subclass'], legend_title='subclass',
        label_order=[cfg['mouse_subclass'] for cfg in SUBCLASSES], cmap='tab10', s=2)

# ---------------------------------------------------------------------------
# 11. The geometry the archetype centroids form
# ---------------------------------------------------------------------------
P = centroids[list(ARC_SPC)]
D = pd.DataFrame([[np.linalg.norm(P.loc[i] - P.loc[j]) for j in ARCH_ORDER]
                  for i in ARCH_ORDER], index=ARCH_ORDER, columns=ARCH_ORDER)

arc, angle, centre, cut_gap = arc_from_centroids(P)
radius     = pd.Series(np.linalg.norm((P - centre).values, axis=1), index=ARCH_ORDER)
med_radius = radius.median()
n_by_arch  = assign['archetype'].value_counts()

print(f'\nArchetype order around the arc in {"-".join(ARC_SPC)}:')
for name in arc:
    flag = ('   INTERIOR — its angle about the centre is not meaningful'
            if radius[name] < INTERIOR_FRAC * med_radius else '')
    print(f'   {name:9s} ' + '  '.join(f'{c}={P.loc[name, c]:7.2f}' for c in ARC_SPC)
          + f'  angle={angle[name]:7.1f}  r={radius[name]:6.2f} '
            f'({radius[name] / med_radius:.2f}x med)  n={int(n_by_arch.get(name, 0)):5d}{flag}')
print('   ' + ' -> '.join(arc))
runs = subclass_runs(arc)
print(f'   subclasses in arc order ({len(runs)} runs; {len(SUBCLASSES)} = each subclass '
      f'contiguous): ' + ' -> '.join(runs))

even_gap = 360.0 / len(ARCH_ORDER)
print(f'   circle cut at the largest angular gap: {cut_gap:.1f} deg '
      f'(even spacing would be {even_gap:.1f})')
if cut_gap < 1.5 * even_gap:
    print(f'   WARNING: no natural opening — the centroids WRAP the centre rather than '
          f'forming an open arc. A closed loop is evidence against a linear depth '
          f'continuum, and the ordering below should be distrusted.')

axis1 = ARC_SPC[0]
ax_order = list(P[axis1].sort_values().index)
mean_rank = lambda o, sub: np.mean([i for i, n in enumerate(o) if subclass_of(n) == sub])
if mean_rank(ax_order, DIR_FIRST) > mean_rank(ax_order, DIR_LAST):
    ax_order = ax_order[::-1]
rho, _ = spearmanr([arc.index(n) for n in ARCH_ORDER], [ax_order.index(n) for n in ARCH_ORDER])
print(f'\n   {axis1}-only order: ' + ' -> '.join(ax_order))
print(f'   Spearman rho(arc order, {axis1}-only order) = {rho:.3f}, '
      f'{len(inverted_pairs(arc, ax_order))} inverted pairs')

# class-imbalance control. Mouse top-cell counts are 300/300/100/100, so unlike human the
# top-cell centroids are not all averages of equally many cells either.
cent_all = pd.DataFrame([spc_df_all[assign['archetype'] == name].mean()
                         for name in ARCH_ORDER], index=ARCH_ORDER)
arc_all, _, _, _ = arc_from_centroids(cent_all[list(ARC_SPC)])
print(f'\n   all-cell (size-weighted) centroid order: ' + ' -> '.join(arc_all))
cyc = cyclic_offset(arc, arc_all)
if cyc is not None:
    k, rev = cyc
    print(f'   vs the top-cell order: THE SAME CYCLE (rotated by {k}'
          f'{", reversed" if rev else ""}) — the ordering is not driven by class imbalance')
else:
    inv_all = inverted_pairs(arc, arc_all)
    print(f'   vs the top-cell order: a different cycle, {len(inv_all)} inverted pairs '
          f'{inv_all}')

pairs  = list(itertools.combinations(ARCH_ORDER, 2))
within = [D.loc[a, b] for a, b in pairs if subclass_of(a) == subclass_of(b)]
cross  = [D.loc[a, b] for a, b in pairs if subclass_of(a) != subclass_of(b)]
print(f'\nCentroid distances: within-subclass median {np.median(within):.2f} (n={len(within)}), '
      f'cross-subclass median {np.median(cross):.2f} (n={len(cross)})')
print(f'  cross pairs closer than the median within-subclass pair: '
      f'{sum(1 for v in cross if v < np.median(within))}/{len(cross)}')
print('  closest pairs:')
for a, b in sorted(pairs, key=lambda p: D.loc[p])[:12]:
    kind = 'WITHIN' if subclass_of(a) == subclass_of(b) else 'cross '
    print(f'   {kind} {a:9s} {b:9s} d={D.loc[a, b]:6.2f}')

cell_angle = np.degrees(np.arctan2(*(spc_df_all[list(ARC_SPC)].values
                                     - centre.values).T[::-1]))
hist, _ = np.histogram(cell_angle, bins=ARC_BINS, range=(-180, 180))
print(f'\nCells per {360 // ARC_BINS}-degree bin around the arc — a continuum has no empty '
      f'bin between occupied ones:')
print('   ' + ' '.join(f'{v}' for v in hist))
interior = hist[np.argmax(hist > 0):len(hist) - np.argmax(hist[::-1] > 0)]
print(f'   occupied span: {len(interior)} bins, minimum {interior.min()} cells, '
      f'{int((interior == 0).sum())} empty')

# ---------------------------------------------------------------------------
# 12. The depth-order label map, and how it compares with script 15's
# ---------------------------------------------------------------------------
if ARC_ORDER_CURATED is None:
    print(UNCURATED_BANNER)
    arc_order, curated = arc, False
elif sorted(ARC_ORDER_CURATED) != sorted(ARCH_ORDER):
    raise ValueError(f'ARC_ORDER_CURATED is not a permutation of the {len(ARCH_ORDER)} '
                     f'archetypes: {sorted(set(ARC_ORDER_CURATED) ^ set(ARCH_ORDER))}')
else:
    print(f'\nARC_ORDER_CURATED is set ({len(ARC_ORDER_CURATED)} labels) — run 2, curated.')
    arc_order, curated = list(ARC_ORDER_CURATED), True

label_meta = {f'{cfg["mouse_subclass"]} {ALPHABET[k]}': (cfg['token'], cfg['mouse_subclass'],
                                                         ALPHABET[k])
              for cfg in SUBCLASSES for k in range(cfg['noc'])}
within_rank, arc_rows = {}, []
for rank, label in enumerate(arc_order):
    token, subclass, old_letter = label_meta[label]
    j = within_rank.get(subclass, 0)
    within_rank[subclass] = j + 1
    arc_rows.append({'key': f'{token}_{old_letter}', 'token': token,
                     'mouse_subclass': subclass, 'old_letter': old_letter,
                     'old_label': label, 'arc_rank': rank,
                     'arc_rank_angular': arc.index(label),
                     'arc_rank_within': j, 'new_letter': f"{ALPHABET[j]}'",
                     'new_label': f"{subclass} {ALPHABET[j]}'", 'curated': curated,
                     'arc_basis': '-'.join(ARC_SPC)})

arc_df = pd.DataFrame(arc_rows)
arc_df.to_csv(OUT_ARC_ORDER, sep='\t', index=False)
print(f'\nSaved {OUT_ARC_ORDER}')
print('  depth order -> display label:')
for r in arc_rows:
    print(f"   {r['arc_rank']:2d}  {r['old_label']:9s} -> {r['new_label']:10s}"
          f"  (angular rank {r['arc_rank_angular']})")

if not curated:
    print('  curated-vs-angular inversions: NOT APPLICABLE (run 1 — arc_rank IS the angular '
          'rank; there is nothing to compare)')
else:
    inverted = inverted_pairs(arc_order, arc)
    print(f'  curated order inverts {len(inverted)} of '
          f'{len(arc_rows) * (len(arc_rows) - 1) // 2} pairs vs the angular sort: '
          f'{inverted if inverted else "none"}')

# --- the control this script exists for: does the pipeline reproduce 15's arc? ---
arc15 = pd.read_csv(IN_ARC_15, sep='\t').sort_values('arc_rank')
order15 = list(arc15['old_label'])
if sorted(order15) != sorted(ARCH_ORDER):
    raise ValueError(f'{IN_ARC_15} does not cover the same {len(ARCH_ORDER)} archetypes')
inv15 = inverted_pairs(arc_order, order15)
cyc15 = cyclic_offset(arc_order, order15)
print(f'\nVersus script 15 (raw PC1-PC2, curated):')
print('   15:   ' + ' -> '.join(order15))
print('   this: ' + ' -> '.join(arc_order))
print(f'   {len(inv15)} of {len(pairs)} pairs inverted; subclass runs '
      f'{" -> ".join(subclass_runs(order15))} (15) vs {" -> ".join(subclass_runs(arc_order))}')
if cyc15 is not None:
    k, rev = cyc15
    print(f'   the two are THE SAME CYCLE (rotated by {k}{", reversed" if rev else ""})')
print('   NOTE 15\'s order is curated (its L6IT segment was corrected by inspection), so a '
      'disagreement here is not automatically this script\'s error.')

print('\nDone.  ARC ORDER STATUS: '
      + ('UNCURATED (run 1) — set ARC_ORDER_CURATED and rerun.' if not curated
         else 'CURATED (run 2).'))
