"""Project the whole yoo25 NR developmental series into the adult (P21) PC space, per subclass.

Script 61 projects P8 into a PC basis fit on P21 alone. This is the same logic over the FULL
normally-reared series -- P6, P8, P10, P12, P14, P17, P21 -- so the approach to the adult
archetype structure can be read as a trajectory rather than a single before/after contrast.

The dark-reared ages (P12DR, P14DR, P17DR, P21DR) are deliberately excluded: this script asks a
developmental question, and mixing a rearing manipulation into the age axis would confound it.

As in 61, Harmony is sidestepped entirely. Every age comes from the SAME yoo25 experiment, so
there is no cross-dataset batch effect for Harmony to remove, and a PCA refit on P21 alone gives
an exact, reproducible basis. Every statistic -- HVG selection, the z-score mean/sd, the
library-size regression, the PCA -- is fit on P21 and APPLIED to the earlier ages. The P21 basis
never moves, so all seven ages land in one fixed adult coordinate system.

The marker-score clip is pooled across all seven ages so the score VALUES are comparable between
timepoints. Panel COLOUR, by contrast, is blended within each age, so every panel spends its full
colour range on the spread that actually exists there. The trajectory figure keeps a second,
pooled blend: a per-age rescale would re-centre each timepoint on its own mean and flatten the
very drift the trajectory exists to plot.

Procedure (per subclass; 61's steps, with the query set widened from one age to six):
  1) subset yoo25 AllAges to the subclass, keep the NR ages; reference = P21
  2) HVG on P21 only (top 2000 by variance, common.select_hvg)
  3) CP10k -> log2(1+x); z-score with P21's per-gene mean/sd, applied to every age
  4) regress out log library size, model fit on P21 and applied to every age
  5) PCA(10) fit on P21; transform every age
  6) per-cell archetype scores from the 33-36 marker gene sets, clip statistics pooled over
     all NR ages
  7) colour = script 59's continuous score blend, rescaled WITHIN each age for the panels and
     pooled over all ages for the trajectory. An argmax assignment is written to the score TSV
     but does NOT drive any figure -- argmax is a hard call that can zero out an archetype whose
     markers merely shift down with age.
  8) overlay the P21 archetype simplex fit by script 62, on the P21 panel only and all in
     black -- it is a reference frame, not one of the coloured score channels

Caveats (same as 61):
  - No batch correction anywhere. Each age is its own set of replicates (P6a/b/c ... P21a/b), so
    the age trajectory is confounded with any age-specific technical batch effect. This is the
    price of keeping the adult basis fixed, and it is the honest trade: correcting age as a
    batch would remove the signal.
  - This PC basis is NEW. It is not the 33-36 / 59 coordinate system: do not compare these PC
    values against *_pcha_xp.tsv, and do not overlay the *_pcha_aa.tsv vertices, which live in
    the Harmony-derived space. The simplex drawn here is a SEPARATE PCHA fit on P21 alone
    (script 62), expressed in these PC units, and it is shown on the P21 panel only -- drawing
    it over the younger ages would imply an archetype structure that was never fit there.
  - Panels show only their own timepoint's cells. Axis limits are still shared across each row,
    so panels stay comparable; nothing is cropped out.
  - Archetype identity enters only through the marker gene sets, which were called on the
    two-dataset Harmony fit.
  - Early ages sitting outside the P21 cloud, or showing less variance along PC1/PC2, is the
    developmental signal, not a bug.

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv
  local_data/res/it/3X.follow.two_*_archetype_scores.tsv        (validation only)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
  local_data/res/it/62.yoo25_p21_<token>_archetype_coords.tsv   (P21 archetype vertices)
Run order:
  This script -> 62 -> this script. The first pass emits the PC coords 62 fits its archetypes in;
  62 writes the vertices; the second pass draws them. A missing 62 file stops the script.
Outputs:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
  local_data/res/it/61.v2.yoo25_p21_<token>_pc_loadings.tsv
  local_data/res/it/61.v2.yoo25_nr_<token>_archetype_scores.tsv
  local_data/res/it/61.v2.yoo25_p21_pc_variance.tsv
  local_data/res/it/61.v2.yoo25_nr_archetype_weight_by_age.tsv
  local_data/fig/it/61.v2.yoo25_nr_ages_projected_to_p21_pcs.pdf
  local_data/fig/it/61.v2.yoo25_nr_archetype_weight_by_age.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from common import select_hvg

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_H5AD       = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_AllAges.h5ad')
IN_ARCMAP     = os.path.join(IT_EVO_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
OUT_COORDS    = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_pc_coords.tsv')
OUT_LOADINGS  = os.path.join(RES_DIR, '61.v2.yoo25_p21_{token}_pc_loadings.tsv')
OUT_SCORES    = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_archetype_scores.tsv')
OUT_VARIANCE  = os.path.join(RES_DIR, '61.v2.yoo25_p21_pc_variance.tsv')
OUT_TRAJ_TSV  = os.path.join(RES_DIR, '61.v2.yoo25_nr_archetype_weight_by_age.tsv')
OUT_PDF       = os.path.join(FIG_DIR, '61.v2.yoo25_nr_ages_projected_to_p21_pcs.pdf')
OUT_TRAJ_PDF  = os.path.join(FIG_DIR, '61.v2.yoo25_nr_archetype_weight_by_age.pdf')
# Archetype vertices fit on P21 alone by script 62, in THIS script's PC units. Bootstrap order is
# 61.v2 -> 62 -> 61.v2: the first pass writes the coords 62 needs, 62 fits the vertices, the
# second pass draws them. Missing file is a hard stop, not a silently skipped overlay.
IN_ARCH       = os.path.join(RES_DIR, '62.yoo25_p21_{token}_archetype_coords.tsv')

# `token` keys into the depth-arc table; `stem` prefixes the cached 33-36 marker/score TSVs.
SUBCLASSES = [
    dict(subclass='L2/3', token='L23',  sub_val='L2/3', stem='34.follow.two_L23'),
    dict(subclass='L4',   token='L4',   sub_val='L4',   stem='33.follow.two_L4'),
    dict(subclass='L5IT', token='L5IT', sub_val='L5IT', stem='35.follow.two_L5IT'),
    dict(subclass='L6IT', token='L6IT', sub_val='L6IT', stem='36.follow.two_L6IT'),
]

# --- parameters ---
AGE_COL        = 'Age'
SUBCLASS_COL   = 'Subclass'
SAMPLE_COL     = 'Sample'
TYPE_COL       = 'Type'
DEPTH_COL      = 'total_counts'
# Normally-reared series only, youngest first. The DR ages are a rearing manipulation, not an
# age, and would confound the developmental axis.
AGES           = ['P6', 'P8', 'P10', 'P12', 'P14', 'P17', 'P21']
REF_AGE        = 'P21'
N_HVG          = 2000
N_PCS          = 10
CP10K          = 1e4
SCORE_PCTILE   = (2, 98)          # matches step 6 of the 33-36 follow scripts
ARCH_LETTERS   = ['A', 'B', 'C', 'D', 'E', 'F']   # archetype_1 -> A, archetype_2 -> B, ...
# Colour follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS    = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
VALIDATE_R_MIN = 0.99             # recomputed vs persisted P21 scores (differ only in clip set)
BLEND_PCTILE   = (5, 95)          # per-score clip before rescaling (59's choice)
PALE_RGB       = np.array(mcolors.to_rgb('#e6e6e6'))   # cell with no dominant archetype
MIN_MIX        = 0.12             # floor on the pale->hue mix, so the palest cells still tint
GAMMA          = 0.5              # <1 lifts middling margins out of the gray
POINT_SIZE     = 2.5
FIG_PANEL_W    = 2.7
FIG_PANEL_H    = 2.7
TRAJ_PANEL_W   = 3.2    # trajectory: four subclass panels in a row, shared y axis
TRAJ_PANEL_H   = 3.0
DPI            = 300


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


def cp10k_log2(counts, depth):
    """CP10k -> log2(1+x), the normalization shared by 33-36 and 52/53."""
    return np.log2(counts / depth[:, None] * CP10K + 1.0)


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def clip_scale(mat):
    """Per-gene percentile clip -> min-max into [0, 1] (33-36 step 6 / 53's recipe)."""
    lo, hi = np.percentile(mat, SCORE_PCTILE, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((mat - lo) / rng, 0.0, 1.0)


def blend(scores, rgb_by_col):
    """Continuous archetype-score blend -> (rgb, weights, conf). Verbatim rule of script 59.

    Each score column is clipped at its own BLEND_PCTILE and rescaled to [0, 1], the rescaled
    values are normalised to sum to 1 (a point on the archetype simplex), and the hue is that
    convex mix of the archetype colours. Saturation follows the winning margin, so ambiguous
    cells read pale and only committed cells carry saturated colour.
    """
    lo, hi = np.percentile(scores, BLEND_PCTILE, axis=0)
    if np.any(hi <= lo):
        raise ValueError(f'score column with no spread at percentiles {BLEND_PCTILE}: '
                         f'lo={lo}, hi={hi}')
    w = np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
    tot = w.sum(axis=1, keepdims=True)
    k = scores.shape[1]
    # A cell under the low percentile on EVERY archetype has no weight at all; it is maximally
    # ambiguous by construction, so it takes the uniform point and renders palest.
    weights = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 1.0 / k)
    hue = weights @ np.asarray(rgb_by_col)
    conf = (weights.max(axis=1) - 1.0 / k) / (1.0 - 1.0 / k)   # 0 = uniform, 1 = one-hot
    mix = MIN_MIX + (1.0 - MIN_MIX) * conf ** GAMMA
    return PALE_RGB + (hue - PALE_RGB) * mix[:, None], weights, conf


print(f'--- yoo25 NR series {AGES} projected into the {REF_AGE} PC space, per IT subclass ---')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

if REF_AGE not in AGES:
    raise ValueError(f'REF_AGE {REF_AGE} must be one of AGES {AGES}')
for cfg in SUBCLASSES:                       # fail fast, before any compute
    for suffix in ('archetype_markers', 'archetype_scores'):
        path = os.path.join(RES_DIR, f'{cfg["stem"]}_{suffix}.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path}')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')
for cfg in SUBCLASSES:
    path = IN_ARCH.format(token=cfg['token'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{cfg["subclass"]}: missing {path}. Run scripts/it/62 first -- it fits the P21 '
            f'archetypes in this script\'s PC units. On a cold start run this script once to '
            f'emit the coords 62 needs, then 62, then this script again.')

relabel_by_token = load_relabel()

# ===================== load once, keep only the NR ages =====================

print(f'Loading {IN_H5AD}...')
adata_all = ad.read_h5ad(IN_H5AD)
age_all = adata_all.obs[AGE_COL].astype(str)
absent_ages = sorted(set(AGES) - set(age_all.unique()))
if absent_ages:
    raise ValueError(f'{IN_H5AD}: requested age(s) {absent_ages} not in {AGE_COL}')
adata_all = adata_all[age_all.isin(AGES).values].copy()
print(f'  kept {adata_all.n_obs} cells over {len(AGES)} NR ages, {adata_all.n_vars} genes')

raw_genes = np.array(adata_all.raw.var_names)
gene_keep = ~np.char.startswith(np.char.lower(raw_genes.astype(str)), 'mt-')
genes = raw_genes[gene_keep]
print(f'  gene universe: {len(genes)} ({(~gene_keep).sum()} mito genes removed)')
gene_to_col = {g: i for i, g in enumerate(raw_genes)}

panels, variance_rows, traj_rows = [], [], []

for cfg in SUBCLASSES:
    S, token, stem = cfg['subclass'], cfg['token'], cfg['stem']
    print(f'\n=== {S} ===')

    # --- 1. subset to the subclass; the reference is one age, the query is all the others ---
    sub_mask = (adata_all.obs[SUBCLASS_COL].astype(str) == cfg['sub_val']).values
    a = adata_all[sub_mask]
    ages = a.obs[AGE_COL].astype(str).values
    n_by_age = {age: int((ages == age).sum()) for age in AGES}
    empty = [age for age, n in n_by_age.items() if n == 0]
    if empty:
        raise ValueError(f'{S}: no cells at age(s) {empty}')
    ref_mask = ages == REF_AGE
    n_ref = n_by_age[REF_AGE]
    print('  cells: ' + '  '.join(f'{age} {n_by_age[age]}' for age in AGES))

    depth = a.obs[DEPTH_COL].values.astype(np.float64)   # true library size, pre gene subsetting
    if not (np.all(np.isfinite(depth)) and np.all(depth > 0)):
        raise ValueError(f"{S}: invalid depth column '{DEPTH_COL}' (NaN or <=0)")

    # --- 2. HVG on the REFERENCE only: P21 defines the basis ---
    x_ref_full = dense(a.raw[:, genes].X[ref_mask]).astype(np.float64)
    hvg_mask = select_hvg(x_ref_full, depth[ref_mask], N_HVG)
    hvg_genes = genes[hvg_mask]
    print(f'  HVG: {hvg_mask.sum()} genes selected on {REF_AGE} alone')

    x_hvg = dense(a.raw[:, hvg_genes].X).astype(np.float64)
    del x_ref_full

    # --- 3. CP10k -> log2, then z-score with REFERENCE mean/sd applied to EVERY age ---
    l2 = cp10k_log2(x_hvg, depth)
    mu, sd = l2[ref_mask].mean(axis=0), l2[ref_mask].std(axis=0)   # ddof=0, matches zscore
    if not np.all(sd > 0):
        raise ValueError(f'{S}: {int((sd <= 0).sum())} HVG(s) have zero variance at {REF_AGE}')
    xn = (l2 - mu) / sd
    del l2, x_hvg

    # --- 4. regress out log library size; model fit on the REFERENCE, applied to EVERY age ---
    log_depth = np.log(depth).reshape(-1, 1)
    reg = LinearRegression().fit(log_depth[ref_mask], xn[ref_mask])
    xn = xn - reg.predict(log_depth)

    # --- 5. PCA fit on the REFERENCE, every age transformed ---
    pca = PCA(N_PCS, random_state=0).fit(xn[ref_mask])
    pcs = pca.transform(xn)
    evr = pca.explained_variance_ratio_
    print(f'  PCA fit on {REF_AGE}: PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%, '
          f'top-{N_PCS} {evr.sum()*100:.1f}% of variance')
    del xn

    pc_cols = [f'PC{i+1}' for i in range(N_PCS)]
    coords = pd.DataFrame(pcs, index=a.obs_names.values, columns=pc_cols)
    coords[AGE_COL]    = ages
    coords[SAMPLE_COL] = a.obs[SAMPLE_COL].astype(str).values
    coords[TYPE_COL]   = a.obs[TYPE_COL].astype(str).values
    coords.to_csv(OUT_COORDS.format(token=token), sep='\t')
    pd.DataFrame(pca.components_.T, index=hvg_genes,
                 columns=pc_cols).to_csv(OUT_LOADINGS.format(token=token), sep='\t')
    variance_rows += [dict(subclass=S, PC=c, explained_variance_ratio=v)
                      for c, v in zip(pc_cols, evr)]

    # --- 6. archetype scores from the 33-36 marker sets, clip pooled over ALL NR ages ---
    mk = pd.read_csv(os.path.join(RES_DIR, f'{stem}_archetype_markers.tsv'), sep='\t')
    arch_ids = sorted(mk['archetype'].unique(), key=lambda s: int(s.split('_')[1]))
    letters = [ARCH_LETTERS[int(s.split('_')[1]) - 1] for s in arch_ids]    # NOC derived, not fixed
    relabel = relabel_by_token[token]
    missing_letters = set(letters) - set(relabel)
    if missing_letters:
        raise ValueError(f'{S}: depth arc has no entry for {sorted(missing_letters)}')

    marker_sets = {L: mk.loc[mk['archetype'] == aid, 'gene'].values
                   for L, aid in zip(letters, arch_ids)}
    absent = sorted({g for gs in marker_sets.values() for g in gs if g not in gene_to_col})
    if absent:
        raise ValueError(f'{S}: {len(absent)} marker gene(s) absent from raw.var_names, '
                         f'e.g. {absent[:5]}')

    scores = np.zeros((a.n_obs, len(letters)), dtype=np.float64)
    for k, L in enumerate(letters):
        cols = [gene_to_col[g] for g in marker_sets[L]]
        mat = cp10k_log2(dense(a.raw.X[:, cols]).astype(np.float64), depth)
        scores[:, k] = clip_scale(mat).mean(axis=1)
        print(f'  score_{L} ({relabel[L]}): {len(cols)} marker genes')

    assigned = np.array([letters[i] for i in scores.argmax(axis=1)])
    scores_df = pd.DataFrame(scores, index=a.obs_names.values,
                             columns=[f'score_{L}' for L in letters])
    scores_df[AGE_COL]           = ages
    scores_df['assigned']        = assigned
    scores_df['assigned_primed'] = np.array([relabel[L] for L in assigned])
    scores_df.to_csv(OUT_SCORES.format(token=token), sep='\t')

    # --- cross-check: recomputed P21 scores vs the persisted two-dataset scores ---
    persisted = pd.read_csv(os.path.join(RES_DIR, f'{stem}_archetype_scores.tsv'),
                            sep='\t', index_col=0)
    persisted = persisted[persisted.index.str.startswith('yoo25:')]
    persisted.index = persisted.index.str.replace('yoo25:', '', regex=False)
    shared = persisted.index.intersection(scores_df.index[ref_mask])
    if len(shared) != len(persisted):
        raise ValueError(f'{S}: {len(persisted) - len(shared)} persisted yoo25 cell(s) missing '
                         f'from the {REF_AGE} subset -- the reference is not the 33-36 yoo25 side')
    for L in letters:
        r = np.corrcoef(scores_df.loc[shared, f'score_{L}'],
                        persisted.loc[shared, f'score_{L}'])[0, 1]
        print(f'  validate score_{L} vs persisted ({len(shared)} cells): r = {r:.4f}')
        if r < VALIDATE_R_MIN:
            raise ValueError(f'{S}: recomputed score_{L} correlates r={r:.4f} with the persisted '
                             f'adult score, below {VALIDATE_R_MIN} -- marker-score recipe drifted')

    # --- 7. colour. Two blends, because the two figures ask different questions. ---
    # PANEL colour is rescaled WITHIN each timepoint, so every panel spends its full colour range
    # on the spread that actually exists at that age.
    # TRAJECTORY weights stay pooled over all ages: a per-age rescale would re-centre every
    # timepoint on its own mean and flatten the very drift the trajectory plots.
    rgb_by_col = [np.array(mcolors.to_rgb(ARCH_COLORS[relabel[L]])) for L in letters]
    _, weights, conf = blend(scores, rgb_by_col)              # pooled -> trajectory + TSV
    rgb = np.zeros((a.n_obs, 3), dtype=np.float64)
    for age in AGES:                                          # per age -> panel colour
        m = ages == age
        rgb[m], _, _ = blend(scores[m], rgb_by_col)

    # --- archetype vertices: PCHA on P21 alone, fit by script 62 in these same PC units ---
    arch = pd.read_csv(IN_ARCH.format(token=token), sep='\t', index_col=0)
    if list(arch.index) != letters:
        raise ValueError(f'{S}: {IN_ARCH.format(token=token)} index {list(arch.index)} '
                         f'!= marker-file letters {letters}')
    vertices = arch[['PC1', 'PC2']].values

    traj_rows += [dict(subclass=S, age=age, archetype=relabel[L],
                       n_cells=n_by_age[age],
                       mean_weight=weights[ages == age, k].mean(),
                       argmax_frac=np.mean(assigned[ages == age] == L),
                       median_conf=np.median(conf[ages == age]))
                  for age in AGES for k, L in enumerate(letters)]

    panels.append(dict(subclass=S, pcs=pcs[:, :2], ages=ages, assigned=assigned,
                       letters=letters, relabel=relabel, vertices=vertices, rgb=rgb,
                       weights=weights, conf=conf, evr=evr, n_by_age=n_by_age))

pd.DataFrame(variance_rows).to_csv(OUT_VARIANCE, sep='\t', index=False)
traj = pd.DataFrame(traj_rows)
traj.to_csv(OUT_TRAJ_TSV, sep='\t', index=False)
print(f'\nSaved per-subclass coords/loadings/scores, {OUT_VARIANCE} and {OUT_TRAJ_TSV}')

# ===================== figure 1: subclass (row) x age (col) embedding grid =====================

# One subclass per ROW, ages running left to right, so each row reads as that subclass's
# developmental trajectory and the shared per-row axis limits are immediately obvious.
plt.rcParams['pdf.fonttype'] = 42            # editable vector text
fig, axes = plt.subplots(len(panels), len(AGES), squeeze=False,
                         figsize=(FIG_PANEL_W * len(AGES), FIG_PANEL_H * len(panels)))

for row, P in enumerate(panels):
    xy = P['pcs']
    # Limits span all cells AND the vertices, shared by every panel in the row. Kept separate
    # from `xy`, which stays cell-indexed for the per-age masks below.
    extent = np.vstack([xy, P['vertices']])
    pad = 0.04 * (extent.max(axis=0) - extent.min(axis=0))
    xlim = (extent[:, 0].min() - pad[0], extent[:, 0].max() + pad[0])
    ylim = (extent[:, 1].min() - pad[1], extent[:, 1].max() + pad[1])

    for col, age in enumerate(AGES):
        ax = axes[row][col]
        # Only this timepoint's cells are drawn. The axis limits are still shared across the
        # whole row, so the panels remain directly comparable without a background cloud.
        this = P['ages'] == age
        ax.scatter(xy[this, 0], xy[this, 1], s=POINT_SIZE, c=P['rgb'][this],
                   linewidths=0, zorder=2, rasterized=True)

        # The archetype simplex is drawn ONLY on the P21 panel, because that is the only age it
        # was fit on -- repeating it over the younger panels would imply an archetype structure
        # that was never estimated there. Row axis limits still include the vertices, so the
        # earlier panels stay on the P21 panel's coordinates and remain readable against it.
        # All black: the simplex is a reference frame, not one of the coloured score channels.
        if age == REF_AGE:
            vx = P['vertices']
            closed = np.vstack([vx, vx[0]]) if len(vx) > 2 else vx   # NOC=2 is a segment
            ax.plot(closed[:, 0], closed[:, 1], '-', color='black', linewidth=0.9, zorder=3)
            ax.scatter(vx[:, 0], vx[:, 1], marker='D', c='black', s=42, zorder=4)
            for (ax_, ay_), L in zip(vx, P['letters']):
                ax.annotate(P['relabel'][L], (ax_, ay_), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, fontweight='bold', color='black', zorder=5)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if row == len(panels) - 1:
            ax.set_xlabel('PC1')
        if col == 0:
            # Each ROW is its own P21 basis, so the variance shares belong to the row label --
            # an xlabel would only ever report the bottom row's subclass.
            ax.set_ylabel(f'{P["subclass"]}\n'
                          f'PC1 {P["evr"][0]*100:.1f}% · PC2 {P["evr"][1]*100:.1f}%',
                          fontweight='bold')
        basis = ' (basis)' if age == REF_AGE else ''
        ax.set_title(f'{age}  n={P["n_by_age"][age]}{basis}', fontsize=9)
        sns.despine(ax=ax)   # no colour key: each archetype's hue sits at its labelled anchor

fig.suptitle(f'yoo25 IT NR series projected into the {REF_AGE} PC basis (no Harmony; colour = '
             f'archetype-score blend rescaled within each age; simplex = {REF_AGE}-only PCHA fit, script 62)',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'Saved {OUT_PDF}')

# ===================== figure 2: archetype weight trajectories =====================

# Four panels in a ROW on one shared y scale, so weights are directly comparable between
# subclasses. Note the uniform point differs by NOC -- 1/2 for L5IT, 1/3 for the rest -- so each
# panel keeps its own dotted reference line even though the axis is shared.
fig, axes = plt.subplots(1, len(panels), squeeze=False,
                         figsize=(TRAJ_PANEL_W * len(panels), TRAJ_PANEL_H), sharey=True)
x = np.arange(len(AGES))
for col, (ax, P) in enumerate(zip(axes[0], panels)):
    sub = traj[traj['subclass'] == P['subclass']]
    for L in P['letters']:
        primed = P['relabel'][L]
        y = [sub[(sub['age'] == age) & (sub['archetype'] == primed)]['mean_weight'].iloc[0]
             for age in AGES]
        ax.plot(x, y, '-o', color=ARCH_COLORS[primed], markersize=4, label=primed)
    ax.axhline(1.0 / len(P['letters']), color='0.7', linestyle=':', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(AGES, rotation=45)
    ax.set_xlabel('age (NR)')
    if col == 0:
        ax.set_ylabel('mean blend weight')
    ax.set_title(f'{P["subclass"]}  (NOC={len(P["letters"])})')
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)

fig.suptitle('Adult archetype weight across the yoo25 NR series '
             '(dotted line = the uniform point for that NOC, no archetype preference)',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT_TRAJ_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'Saved {OUT_TRAJ_PDF}')

# ===================== summary =====================

print('\n--- archetype composition by age: mean blend weight / argmax fraction ---')
print('    The blend weights drive the panel colour and degrade gracefully; the argmax fraction')
print('    is a hard call that can zero out an archetype whose markers merely shift down.')
for P in panels:
    print(f'  {P["subclass"]}')
    for age in AGES:
        m = P['ages'] == age
        cols = ', '.join(
            f'{P["relabel"][L]} {P["weights"][m, k].mean():.2f}/{np.mean(P["assigned"][m] == L):.2f}'
            for k, L in enumerate(P['letters']))
        print(f'    {age:4s} n={P["n_by_age"][age]:5d}  {cols}'
              f'   [margin conf median {np.median(P["conf"][m]):.2f}]')

print('\nDone.')
