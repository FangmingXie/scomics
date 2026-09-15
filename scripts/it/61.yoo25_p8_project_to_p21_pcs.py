"""Project P8 IT cells into the adult (P21) PC space, one basis per subclass — yoo25 only.

Scripts 33-36 define the IT archetype structure on a two-dataset Harmony embedding
(cheng22 P28NR + yoo25 P21), and script 59 renders each layer in the plane of that embedding.
This script asks where P8 cells sit relative to that adult structure.

P8 cannot be routed through the 33-36 basis: harmonypy has no transform() for new cells, no
Harmony state is persisted, and the z-score / depth-regression / PCA statistics behind the
adult fit were never saved either. So Harmony is sidestepped entirely. P21 and P8 both come
from the SAME yoo25 experiment, so there is no cross-dataset batch effect for Harmony to
remove in the first place: a PCA refit on P21 alone gives an exact, fully reproducible basis,
and P8 becomes a true projection into it rather than an estimate.

Every statistic -- HVG selection, the z-score mean/sd, the library-size regression, the PCA --
is fit on P21 and APPLIED to P8. The P21 basis never moves.

Archetype identity is carried over by the marker-score route (the scripts 52/53 recipe), which
needs no geometry in the new space and therefore no refit of PCHA.

Procedure (per subclass; mirrors 3X.harmony.two_*_embed.py minus the merge and minus Harmony):
  1) subset yoo25 AllAges to the subclass; reference = Age P21, query = Age P8
  2) HVG on P21 only (top 2000 by variance, common.select_hvg)
  3) CP10k -> log2(1+x); z-score with P21's per-gene mean/sd, applied to both ages
  4) regress out log library size, model fit on P21 and applied to P8
  5) PCA(10) fit on P21; transform both ages
  6) per-cell archetype scores from the 33-36 marker gene sets, clip statistics pooled over
     P21+P8 so the two ages share one scale
  7) panel colour = script 59's continuous score blend, its rescale ALSO pooled over both ages.
     An argmax assignment is still written to the score TSV, but it does not drive the figure:
     argmax is a hard call that can zero out an archetype whose markers merely shift down,
     which is exactly the failure mode a developmental comparison invites.

Caveats:
  - No batch correction anywhere. P21a/P21b and P8a/P8b/P8c replicate variation is left in, so
    the P8-vs-P21 shift is confounded with any P8-specific technical batch effect.
  - This PC basis is NEW. It is not the 33-36 / 59 coordinate system: do not compare these PC
    values against *_pcha_xp.tsv, and do not overlay the *_pcha_aa.tsv vertices, which live in
    the Harmony-derived space. The diamonds on the figure are ANCHORS -- centroids of the
    top-decile P21 cells for each archetype score -- not PCHA vertices.
  - Archetype identity enters only through the marker gene sets, which were called on the
    two-dataset Harmony fit.
  - P8 cells sitting outside the P21 cloud, or showing less variance along PC1/PC2, is the
    developmental signal, not a bug.

Reads:
  links/it/superdupermegaRNA_yoo25_IT_AllAges.h5ad
  local_data/res/it/3X.follow.two_*_archetype_markers.tsv
  local_data/res/it/3X.follow.two_*_archetype_scores.tsv        (validation only)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/res/it/61.yoo25_p8_p21_<token>_pc_coords.tsv
  local_data/res/it/61.yoo25_p21_<token>_pc_loadings.tsv
  local_data/res/it/61.yoo25_p8_p21_<token>_archetype_scores.tsv
  local_data/res/it/61.yoo25_p21_pc_variance.tsv
  local_data/fig/it/61.yoo25_p8_projected_to_p21_pcs.pdf
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
OUT_COORDS    = os.path.join(RES_DIR, '61.yoo25_p8_p21_{token}_pc_coords.tsv')
OUT_LOADINGS  = os.path.join(RES_DIR, '61.yoo25_p21_{token}_pc_loadings.tsv')
OUT_SCORES    = os.path.join(RES_DIR, '61.yoo25_p8_p21_{token}_archetype_scores.tsv')
OUT_VARIANCE  = os.path.join(RES_DIR, '61.yoo25_p21_pc_variance.tsv')
OUT_PDF       = os.path.join(FIG_DIR, '61.yoo25_p8_projected_to_p21_pcs.pdf')

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
REF_AGE        = 'P21'
QUERY_AGE      = 'P8'
N_HVG          = 2000
N_PCS          = 10
CP10K          = 1e4
SCORE_PCTILE   = (2, 98)          # matches step 6 of the 33-36 follow scripts
ARCH_LETTERS   = ['A', 'B', 'C', 'D', 'E', 'F']   # archetype_1 -> A, archetype_2 -> B, ...
# Colour follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS    = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
VALIDATE_R_MIN = 0.99             # recomputed vs persisted P21 scores (differ only in clip set)
# Panel colour is the continuous score blend of script 59, NOT an argmax call. Its rescale runs
# over the POOLED P21+P8 cells so both rows share one colour scale and the age shift is visible;
# per-age rescaling would renormalize P8 back onto the adult palette and hide it.
BLEND_PCTILE   = (5, 95)          # per-score clip before rescaling (59's choice)
PALE_RGB       = np.array(mcolors.to_rgb('#e6e6e6'))   # cell with no dominant archetype
MIN_MIX        = 0.12             # floor on the pale->hue mix, so the palest cells still tint
GAMMA          = 0.5              # <1 lifts middling margins out of the gray
ANCHOR_PCTILE  = 90               # anchor = centroid of the top-decile P21 cells for that score
BG_COLOR       = '#e2e2e2'
POINT_SIZE     = 4
FIG_PANEL_W    = 4.4
FIG_PANEL_H    = 4.4
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


print('--- P8 projected into the adult (P21) PC space, per IT subclass (yoo25 only) ---')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:                       # fail fast, before any compute
    for suffix in ('archetype_markers', 'archetype_scores'):
        path = os.path.join(RES_DIR, f'{cfg["stem"]}_{suffix}.tsv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path}')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')

relabel_by_token = load_relabel()

# ===================== load once, keep only the two ages =====================

print(f'Loading {IN_H5AD}...')
adata_all = ad.read_h5ad(IN_H5AD)
keep_age = adata_all.obs[AGE_COL].astype(str).isin([REF_AGE, QUERY_AGE]).values
adata_all = adata_all[keep_age].copy()
print(f'  kept {adata_all.n_obs} cells at {REF_AGE}/{QUERY_AGE}, {adata_all.n_vars} genes')

raw_genes = np.array(adata_all.raw.var_names)
gene_keep = ~np.char.startswith(np.char.lower(raw_genes.astype(str)), 'mt-')
genes = raw_genes[gene_keep]
print(f'  gene universe: {len(genes)} ({(~gene_keep).sum()} mito genes removed)')
gene_to_col = {g: i for i, g in enumerate(raw_genes)}

panels, variance_rows = [], []

for cfg in SUBCLASSES:
    S, token, stem = cfg['subclass'], cfg['token'], cfg['stem']
    print(f'\n=== {S} ===')

    # --- 1. subset to the subclass and split reference / query ---
    sub_mask = (adata_all.obs[SUBCLASS_COL].astype(str) == cfg['sub_val']).values
    a = adata_all[sub_mask]
    ages = a.obs[AGE_COL].astype(str).values
    ref_mask, qry_mask = ages == REF_AGE, ages == QUERY_AGE
    n_ref, n_qry = int(ref_mask.sum()), int(qry_mask.sum())
    if n_ref == 0 or n_qry == 0:
        raise ValueError(f'{S}: empty age group ({REF_AGE}: {n_ref}, {QUERY_AGE}: {n_qry})')
    print(f'  {REF_AGE}: {n_ref} cells   {QUERY_AGE}: {n_qry} cells')

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

    # --- 3. CP10k -> log2, then z-score with REFERENCE mean/sd applied to BOTH ages ---
    l2 = cp10k_log2(x_hvg, depth)
    mu, sd = l2[ref_mask].mean(axis=0), l2[ref_mask].std(axis=0)   # ddof=0, matches zscore
    if not np.all(sd > 0):
        raise ValueError(f'{S}: {int((sd <= 0).sum())} HVG(s) have zero variance at {REF_AGE}')
    xn = (l2 - mu) / sd
    del l2, x_hvg

    # --- 4. regress out log library size; model fit on the REFERENCE, applied to BOTH ---
    log_depth = np.log(depth).reshape(-1, 1)
    reg = LinearRegression().fit(log_depth[ref_mask], xn[ref_mask])
    xn = xn - reg.predict(log_depth)

    # --- 5. PCA fit on the REFERENCE, both ages transformed ---
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

    # --- 6. archetype scores from the 33-36 marker sets, clip pooled over P21+P8 ---
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

    # --- panel colour: continuous blend over the POOLED cells, so both rows share one scale ---
    rgb_by_col = [np.array(mcolors.to_rgb(ARCH_COLORS[relabel[L]])) for L in letters]
    rgb, weights, conf = blend(scores, rgb_by_col)

    # --- anchors: centroid of the top-decile REFERENCE cells for each score ---
    # The score analogue of 33-36's "N cells nearest each archetype". These are landmarks in a
    # NEW basis, not PCHA vertices -- *_pcha_aa.tsv lives in the Harmony space and cannot come here.
    anchors = np.stack([
        pcs[ref_mask & (scores[:, k] >= np.percentile(scores[ref_mask, k], ANCHOR_PCTILE))][:, :2]
        .mean(axis=0)
        for k in range(len(letters))])

    panels.append(dict(subclass=S, pcs=pcs[:, :2], ages=ages, assigned=assigned,
                       letters=letters, relabel=relabel, anchors=anchors, rgb=rgb,
                       weights=weights, conf=conf, rgb_by_col=rgb_by_col,
                       evr=evr, n_ref=n_ref, n_qry=n_qry))

pd.DataFrame(variance_rows).to_csv(OUT_VARIANCE, sep='\t', index=False)
print(f'\nSaved per-subclass coords/loadings/scores and {OUT_VARIANCE}')

# ===================== figure =====================

plt.rcParams['pdf.fonttype'] = 42            # editable vector text
fig, axes = plt.subplots(2, len(panels), squeeze=False,
                         figsize=(FIG_PANEL_W * len(panels), FIG_PANEL_H * 2))

for col, P in enumerate(panels):
    xy = P['pcs']
    pad = 0.04 * (xy.max(axis=0) - xy.min(axis=0))
    xlim = (xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
    ylim = (xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])

    for row, age in enumerate([REF_AGE, QUERY_AGE]):
        ax = axes[row][col]
        this, other = P['ages'] == age, P['ages'] != age
        ax.scatter(xy[other, 0], xy[other, 1], s=POINT_SIZE, c=BG_COLOR,
                   linewidths=0, zorder=1, rasterized=True)
        ax.scatter(xy[this, 0], xy[this, 1], s=POINT_SIZE, c=P['rgb'][this],
                   linewidths=0, zorder=2, rasterized=True)

        # P21-derived anchors on BOTH rows, so P8's displacement reads against a fixed landmark.
        # Their faces carry the archetype colour, which is the colour key for the blend.
        ax.scatter(P['anchors'][:, 0], P['anchors'][:, 1], marker='D',
                   c=[ARCH_COLORS[P['relabel'][L]] for L in P['letters']],
                   edgecolors='black', linewidths=1.0, s=55, zorder=4)
        for (ax_, ay_), L in zip(P['anchors'], P['letters']):
            ax.annotate(P['relabel'][L], (ax_, ay_), textcoords='offset points', xytext=(5, 5),
                        fontsize=8, fontweight='bold', color='black', zorder=5)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'PC1 ({P["evr"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({P["evr"][1]*100:.1f}%)')
        n = P['n_ref'] if age == REF_AGE else P['n_qry']
        fitted = 'basis' if age == REF_AGE else 'projected'
        ax.set_title(f'{P["subclass"]}  {age}  (n={n}, {fitted})')
        sns.despine(ax=ax)   # no colour key: each archetype's hue sits at its labelled anchor

fig.suptitle(f'yoo25 IT — {QUERY_AGE} projected into the {REF_AGE} PC basis '
             f'(no Harmony; colour = adult archetype-score blend, rescaled over both ages)')
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'Saved {OUT_PDF}')

# ===================== summary =====================

print('\n--- archetype composition: mean blend weight (continuous) vs argmax fraction ---')
print('    The blend weights drive the panel colour and degrade gracefully; the argmax fraction')
print('    is a hard call that can zero out an archetype whose markers merely shift down.')
for P in panels:
    for age, n in [(REF_AGE, P['n_ref']), (QUERY_AGE, P['n_qry'])]:
        m = P['ages'] == age
        cols = ', '.join(f'{P["relabel"][L]} {P["weights"][m, k].mean():.2f}/{np.mean(P["assigned"][m] == L):.2f}'
                         for k, L in enumerate(P['letters']))
        print(f'  {P["subclass"]:5s} {age:4s} n={n:5d}  {cols}'
              f'   [margin conf median {np.median(P["conf"][m]):.2f}]')

print('\nDone.')
