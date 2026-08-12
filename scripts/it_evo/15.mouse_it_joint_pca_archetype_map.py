"""All mouse IT cells in one PC space, colored by which archetype each cell belongs to.

Are the 11 mouse archetypes (L2/3 A,B,C; L4 A,B,C; L5IT A,B; L6IT A,B,C) 11 independent
programs, or samples of one graded manifold spanning the four layers — so that one archetype
turns gradually into another and the two archetypes at a subclass boundary are nearly the
same point?

Nothing computed so far can answer that. Script 05 fits each subclass separately, so the 11
archetypes share no coordinate system; and 05's markers are one-vs-each *within* a subclass,
which excludes by construction any gene roughly constant across that subclass — which is
what a laminar identity gene is. (Within-subclass marker Jaccard is 0.000 for all 10 pairs,
and canonical depth markers are nearly absent from all 11 sets.) A depth continuum could
dominate these cells and leave no trace in 05's output.

This pools all 11,061 IT cells into one embedding, which sees the between-layer genes the
per-subclass fits discard, and colors each cell by its archetype. Three patterns are then
distinguishable by eye:

  continuum  — one connected cloud, subclasses ordered along a single axis, archetypes
               strung along it in laminar order with boundary archetypes overlapping
  recurring  — archetype coloring runs orthogonal to the subclass axis: the same within-layer
               axis rediscovered four times
  discrete   — four separated clouds with density gaps between them

The embedding is scripts/it/19.cheng22_L23_varimax.py:190-221 with its `Subclass` filter
removed and the varimax rotation dropped — same normalization (G2: log2(CP10k+1) from .raw),
same 2000-HVG-by-variance selection, same StandardScaler + PCA(10, random_state=0). Keeping
the procedure identical means any difference from the per-subclass records is the pooling,
not the method. `common.select_hvg` is deliberately NOT used: it normalizes by median depth
and ranks on log1p, so it would select a different gene set than those records.

Caveats:
  * Pooled HVGs are dominated by between-subclass genes. That is the point here, but it makes
    this a new space, not comparable to the per-subclass varimax records, and it must not be
    used to re-derive archetypes.
  * Cell counts are unbalanced (L4 4732 / L2/3 4044 / L6IT 1436 / L5IT 849), so the PCs are
    weighted toward L4 and L2/3.
  * This script's own figures and TSVs keep 05's A/B/C letters — it is the derivation of
    record and its assignment table is keyed by them. Only the survey figures 12 and 14
    relabel, to the primed A'/B'/C' of ARC_ORDER_CURATED below.
  * PCHA's convex weights S would be the canonical archetype assignment, but 05 does not save
    them. Both proxies are emitted — nearest vertex in that subclass's own PCHA space, and
    argmax of 05's scores — plus a top-cells-only label, whose extremes are where the
    archetypes actually are.
  * If the variance partition shows `Sample` dominating PC1-PC2, the follow-up is a
    Harmony-corrected variant; precedent in scripts/it/28-39.harmony.*.

Reads:
  links/it_evo/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  local_data/res/it_evo/05.mouse_<TOKEN>_pcha_{xp,aa}.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_scores.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_top_cells.tsv
Outputs:
  local_data/res/it_evo/15.mouse_IT_joint_pca_{coords,loadings}.tsv
  local_data/res/it_evo/15.mouse_IT_joint_pc_variance_partition.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_assignment.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_centroids.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv   (the depth order and the
      A'/B'/C' display labels that scripts 12 and 14 relabel their mouse columns by; carries
      both the curated rank and this script's angular rank, which disagree on L6IT)
  local_data/fig/it_evo/15.mouse_IT_joint_pca_archetype_map.html
  local_data/fig/it_evo/15.mouse_IT_joint_pca_archetype_{map,subclass}.pdf
"""

import os
import sys
import gc
import itertools
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import scatter_categorical_html, save_archetype_scatter_pdf

# --- file paths ---
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
INPUT_MOUSE   = os.path.join(PROJECT_ROOT, 'links', 'it_evo',
                             'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
OUT_COORDS    = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_pca_coords.tsv')
OUT_LOADINGS  = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_pca_loadings.tsv')
OUT_VAR_PART  = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_pc_variance_partition.tsv')
OUT_ASSIGN    = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_archetype_assignment.tsv')
OUT_CENTROIDS = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_archetype_centroids.tsv')
OUT_HTML      = os.path.join(OUT_FIG_DIR, '15.mouse_IT_joint_pca_archetype_map.html')
OUT_PDF_ARCH  = os.path.join(OUT_FIG_DIR, '15.mouse_IT_joint_pca_archetype_map.pdf')
OUT_PDF_SUB   = os.path.join(OUT_FIG_DIR, '15.mouse_IT_joint_pca_archetype_subclass.pdf')
OUT_ARC_ORDER = os.path.join(OUT_RES_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')

# `noc` must match script 05's noc for the same token.
SUBCLASSES = [
    {'token': 'L23',  'mouse_subclass': 'L2/3', 'noc': 3},
    {'token': 'L4',   'mouse_subclass': 'L4',   'noc': 3},
    {'token': 'L5IT', 'mouse_subclass': 'L5IT', 'noc': 2},
    {'token': 'L6IT', 'mouse_subclass': 'L6IT', 'noc': 3},
]

# --- parameters (N_HVG / N_PCS / column names from it/19) ---
SUBCLASS_COL = 'Subclass'
CLUSTER_COL  = 'Type'
SAMPLE_COL   = 'Sample'
LIBSIZE_COL  = 'n_counts'
N_HVG        = 2000
N_PCS        = 10
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
# The axes the geometry is read on — the cell-type-dominated ones, asserted against the
# variance partition below. PC3 is largely `sample` and must not be used for this.
ARC_PCS      = ('PC1', 'PC2')
ARC_BINS     = 36        # 10-degree bins around the arc, for the connectedness check

# ===========================================================================
# The depth order the survey figures (12, 14) relabel their mouse archetypes by. This is the
# angular ordering computed below WITH THE L6IT SEGMENT CORRECTED by visual inspection of the
# embedding, so it is curated, not purely computed — hence stated here rather than derived.
# The angular sort is unreliable exactly where it was overruled: it split L6IT, putting
# `L6IT A` between L4 and L5IT and ordering C before B, but L6IT B and L6IT C are 5.6 degrees
# and 2.62 units apart (the closest of all 55 centroid pairs, i.e. inside the noise) and
# L6IT A sits in the arc's interior on only 249 cells, where an angle about the centre is
# meaningless. Both ranks are written to OUT_ARC_ORDER so the disagreement stays on record.
# ===========================================================================
ARC_ORDER_CURATED = [
    'L2/3 C', 'L2/3 B', 'L2/3 A',
    'L4 C',   'L4 B',   'L4 A',
    'L5IT B', 'L5IT A',
    'L6IT A', 'L6IT B', 'L6IT C',
]
# ===========================================================================

os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def r2(X, y):
    reg = LinearRegression().fit(X, y)
    ss_res = np.sum((y - reg.predict(X)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def variance_partition(scores, comp_cols, types, samples, libsize, label):
    """Partial R² for Type / sample / library size per component (copied from it/19:96).

    Defined inline in each varimax script rather than in common.py; copied once more here
    instead of refactoring all of them.
    """
    X_type    = pd.get_dummies(types).values.astype(float)
    X_sample  = pd.get_dummies(samples).values.astype(float)
    X_libsize = zscore(libsize).reshape(-1, 1)
    X_full    = np.hstack([X_type, X_sample, X_libsize])

    FACTORS = {'cell_type': X_type, 'sample': X_sample, 'library_size': X_libsize}

    print(f'\nVariance partitioning ({label}):')
    rows = []
    for i, col in enumerate(comp_cols):
        y = scores[:, i]
        r2_full = r2(X_full, y)
        partial = {}
        for name, X_factor in FACTORS.items():
            X_others = np.hstack([v for k, v in FACTORS.items() if k != name])
            partial[name] = max(r2_full - r2(X_others, y), 0.0)
        residual = max(1.0 - r2_full, 0.0)
        rows.append({'component': col, **partial, 'residual': residual})
        print(f"  {col}  R²_full={r2_full:.3f}  type={partial['cell_type']:.3f}  "
              f"sample={partial['sample']:.3f}  libsize={partial['library_size']:.3f}  "
              f"resid={residual:.3f}")
    return pd.DataFrame(rows).set_index('component')


# ---------------------------------------------------------------------------
# 1. All IT cells — the only line that differs from it/19 is the missing Subclass filter
# ---------------------------------------------------------------------------
print(f'Loading {INPUT_MOUSE}...')
m_adata = ad.read_h5ad(INPUT_MOUSE)
print(f'  {m_adata.n_obs} cells x {m_adata.n_vars} genes')

expected = {cfg['mouse_subclass'] for cfg in SUBCLASSES}
present  = set(pd.unique(m_adata.obs[SUBCLASS_COL]))
if present != expected:
    raise ValueError(f'h5ad {SUBCLASS_COL} values {sorted(present)} != the four IT subclasses '
                     f'{sorted(expected)} — the joint embedding assumes the file is exactly '
                     f'the cells the per-subclass records were fit on')
print('  cells per subclass:', m_adata.obs[SUBCLASS_COL].value_counts().to_dict())

types   = m_adata.obs[CLUSTER_COL].values
samples = m_adata.obs[SAMPLE_COL].values
libsize = m_adata.obs[LIBSIZE_COL].values
barcodes = m_adata.obs_names.values

# --- 2. Normalize: raw counts -> log2(CP10k + 1)  (G2, it/19:200-205) ---
print('Normalizing...')
X_raw  = m_adata.raw[:, m_adata.var_names].X.toarray().astype(np.float32)
depths = X_raw.sum(axis=1, keepdims=True)
depths[depths == 0] = 1
X_norm = np.log2(X_raw / depths * 1e4 + 1)
del X_raw
gc.collect()

# --- 3. HVG selection (it/19:207-212) ---
print(f'Selecting top {N_HVG} HVGs across all IT cells...')
gene_var   = X_norm.var(axis=0)
hvg_idx    = np.argsort(gene_var)[::-1][:N_HVG]
gene_names = m_adata.var_names.values[hvg_idx]
X_hvg      = X_norm[:, hvg_idx]
del X_norm
gc.collect()

# --- 4. Scale + PCA (it/19:214-221) ---
print('Scaling and PCA...')
X_scaled = StandardScaler().fit_transform(X_hvg)
pca      = PCA(n_components=N_PCS, random_state=0)
xp       = pca.fit_transform(X_scaled)
pc_cols  = [f'PC{i+1}' for i in range(N_PCS)]

pd.DataFrame(xp, index=barcodes, columns=pc_cols).to_csv(OUT_COORDS, sep='\t')
pd.DataFrame(pca.components_.T, index=gene_names,
             columns=pc_cols).to_csv(OUT_LOADINGS, sep='\t')
print(f'Saved {OUT_COORDS} and {OUT_LOADINGS}')
print('  explained variance ratio: ' +
      ', '.join(f'{c}={v:.3f}' for c, v in zip(pc_cols, pca.explained_variance_ratio_)))
del X_hvg, X_scaled
gc.collect()

# ---------------------------------------------------------------------------
# 5. Archetype assignment per cell, from the per-subclass PCHA records
# ---------------------------------------------------------------------------
print('\nAssigning cells to archetypes...')
blocks = []
for cfg in SUBCLASSES:
    token, subclass, noc = cfg['token'], cfg['mouse_subclass'], cfg['noc']
    xp_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_xp.tsv'),
                            sep='\t', index_col=0)
    aa_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_pcha_aa.tsv'),
                            sep='\t', index_col=0)
    scores_df = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_archetype_scores.tsv'),
                            sep='\t', index_col=0)
    top_df    = pd.read_csv(os.path.join(OUT_RES_DIR, f'05.mouse_{token}_top_cells.tsv'),
                            sep='\t')

    if len(aa_df) != noc:
        raise ValueError(f'{token}: pcha_aa has {len(aa_df)} archetypes but noc={noc}')
    sub_cells = set(m_adata.obs_names[m_adata.obs[SUBCLASS_COL] == subclass])
    if sub_cells != set(xp_df.index):
        raise ValueError(f'{token}: the {len(sub_cells)} {subclass} cells in the h5ad are not '
                         f'the {len(xp_df)} cells of 05.mouse_{token}_pcha_xp.tsv')
    if not scores_df.index.equals(xp_df.index):
        raise ValueError(f'{token}: 05 scores and 05 pcha_xp disagree on cell order')

    names = [f'{subclass} {ALPHABET[k]}' for k in range(noc)]

    # nearest archetype vertex in this subclass's own PCHA space (all PC columns)
    d = np.stack([np.linalg.norm(xp_df.values - aa_df.values[k], axis=1) for k in range(noc)],
                 axis=1)
    nearest = np.array(names)[d.argmin(axis=1)]

    # argmax of 05's own [0,1] scores — an independent proxy for the same thing
    score_cols = [f'score_{ALPHABET[k]}' for k in range(noc)]
    by_score = np.array(names)[scores_df[score_cols].values.argmax(axis=1)]

    # top cells only: where each archetype actually is, everything else greyed out
    top_label = pd.Series('other', index=xp_df.index)
    for k in range(noc):
        cells = top_df[top_df['archetype'] == f'archetype_{k+1}']['cell'].values
        top_label.loc[cells] = names[k]

    blocks.append(pd.DataFrame({'subclass': subclass,
                                'archetype': nearest,
                                'archetype_by_score': by_score,
                                'archetype_top_cells': top_label.values},
                               index=xp_df.index))
    agree = (nearest == by_score).mean()
    print(f'  {token}: {len(xp_df)} cells, nearest-vertex vs score-argmax agree {agree:.1%}')
    print(f'    counts: {pd.Series(nearest).value_counts().to_dict()}')

assign = pd.concat(blocks)
if len(assign) != m_adata.n_obs or not assign.index.is_unique:
    raise ValueError(f'assignment covers {len(assign)} rows ({assign.index.nunique()} unique) '
                     f'but the h5ad has {m_adata.n_obs} cells — every cell must be assigned '
                     f'exactly once')
assign = assign.loc[barcodes]                      # align to the embedding's cell order
assign[CLUSTER_COL] = types
assign.to_csv(OUT_ASSIGN, sep='\t')
print(f'Saved {OUT_ASSIGN}')

ARCH_ORDER = [f'{cfg["mouse_subclass"]} {ALPHABET[k]}'
              for cfg in SUBCLASSES for k in range(cfg['noc'])]

# --- 6. What do the joint PCs encode? Run first: it decides which PCs are interpretable ---
var_df = variance_partition(xp, pc_cols, types, samples, libsize, 'joint IT PCs')
var_df.to_csv(OUT_VAR_PART, sep='\t')
print(f'Saved {OUT_VAR_PART}')

# --- 7. Archetype centroids and the geometry they form ---
xp_df_all = pd.DataFrame(xp, index=barcodes, columns=pc_cols)
centroids = pd.DataFrame(
    [xp_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
centroids.to_csv(OUT_CENTROIDS, sep='\t')
print(f'Saved {OUT_CENTROIDS}')

# Geometry is read on ARC_PCS only. Over all 10 PCs the distances are dominated by the
# sample-driven and near-residual axes (the partition above shows PC3 is mostly `sample`)
# and give a different, misleading ranking of which archetypes are close.
for pc in ARC_PCS:
    if var_df.loc[pc, 'cell_type'] <= var_df.loc[pc, 'sample']:
        raise ValueError(
            f'{pc} is not cell-type dominated (type={var_df.loc[pc, "cell_type"]:.3f} <= '
            f'sample={var_df.loc[pc, "sample"]:.3f}) — ARC_PCS must name the interpretable '
            f'axes, re-read the variance partition before trusting the geometry below')

P = centroids[list(ARC_PCS)]
D = pd.DataFrame([[np.linalg.norm(P.loc[i] - P.loc[j]) for j in ARCH_ORDER]
                  for i in ARCH_ORDER], index=ARCH_ORDER, columns=ARCH_ORDER)
subclass_of = lambda n: n.rsplit(' ', 1)[0]

# order around the arc: a 1-D gradient renders as a curve in 2-D PCA (horseshoe), so the
# ordering along it — not the 2-D spread — is what carries the continuum claim
centre = P.mean()
angle  = pd.Series(np.degrees(np.arctan2(*(P - centre).values.T[::-1])), index=ARCH_ORDER)
arc    = list(angle.sort_values().index)
print(f'\nArchetype order around the arc in {"-".join(ARC_PCS)}:')
for name in arc:
    print(f'   {name:9s} ' + '  '.join(f'{pc}={P.loc[name, pc]:7.2f}' for pc in ARC_PCS) +
          f'  angle={angle[name]:7.1f}')
print('   ' + ' -> '.join(arc))
print('   subclasses in arc order: ' +
      ' -> '.join(pd.Series([subclass_of(n) for n in arc]).drop_duplicates()))

# --- the depth-order label map the survey figures relabel by ---
if sorted(ARC_ORDER_CURATED) != sorted(ARCH_ORDER):
    raise ValueError(f'ARC_ORDER_CURATED is not a permutation of the {len(ARCH_ORDER)} '
                     f'archetypes: {sorted(set(ARC_ORDER_CURATED) ^ set(ARCH_ORDER))}')

label_meta = {f'{cfg["mouse_subclass"]} {ALPHABET[k]}': (cfg['token'], cfg['mouse_subclass'],
                                                         ALPHABET[k])
              for cfg in SUBCLASSES for k in range(cfg['noc'])}
within_rank = {}
arc_rows = []
for rank, label in enumerate(ARC_ORDER_CURATED):
    token, subclass, old_letter = label_meta[label]
    j = within_rank.get(subclass, 0)
    within_rank[subclass] = j + 1
    new_letter = f"{ALPHABET[j]}'"
    arc_rows.append({'key': f'{token}_{old_letter}', 'token': token,
                     'mouse_subclass': subclass, 'old_letter': old_letter,
                     'old_label': label, 'arc_rank': rank,
                     'arc_rank_angular': arc.index(label),
                     'arc_rank_within': j, 'new_letter': new_letter,
                     'new_label': f'{subclass} {new_letter}'})

arc_df = pd.DataFrame(arc_rows)
arc_df.to_csv(OUT_ARC_ORDER, sep='\t', index=False)
print(f'\nSaved {OUT_ARC_ORDER}')
print('  curated depth order -> display label (12 and 14 relabel by this):')
for r in arc_rows:
    print(f"   {r['arc_rank']:2d}  {r['old_label']:9s} -> {r['new_label']:10s}"
          f"  (angular rank {r['arc_rank_angular']})")

# Report inverted PAIRS, not rank shifts: moving one archetype past others renumbers them
# without reordering them relative to each other, and calling that a disagreement would
# overstate how much of the angular result was overruled.
inverted = [(a['old_label'], b['old_label'])
            for a, b in itertools.combinations(arc_rows, 2)
            if a['arc_rank_angular'] > b['arc_rank_angular']]
print(f'  curated order inverts {len(inverted)} of {len(arc_rows) * (len(arc_rows) - 1) // 2} '
      f'pairs vs the angular sort: {inverted if inverted else "none"}')

pairs  = list(itertools.combinations(ARCH_ORDER, 2))
within = [D.loc[a, b] for a, b in pairs if subclass_of(a) == subclass_of(b)]
cross  = [D.loc[a, b] for a, b in pairs if subclass_of(a) != subclass_of(b)]
print(f'\nCentroid distances: within-subclass median {np.median(within):.2f} (n={len(within)}), '
      f'cross-subclass median {np.median(cross):.2f} (n={len(cross)})')
print(f'  cross pairs closer than the median within-subclass pair: '
      f'{sum(1 for v in cross if v < np.median(within))}/{len(cross)}  '
      f'(a continuum whose boundary archetypes collapse would make this large)')
print('  closest pairs:')
for a, b in sorted(pairs, key=lambda p: D.loc[p])[:12]:
    kind = 'WITHIN' if subclass_of(a) == subclass_of(b) else 'cross '
    print(f'   {kind} {a:9s} {b:9s} d={D.loc[a, b]:6.2f}')

# is the cell cloud connected along the arc, or does it break at subclass boundaries?
cell_angle = np.degrees(np.arctan2(*(xp_df_all[list(ARC_PCS)].values - centre.values).T[::-1]))
hist, _ = np.histogram(cell_angle, bins=ARC_BINS, range=(-180, 180))
print(f'\nCells per {360 // ARC_BINS}-degree bin around the arc — a continuum has no empty '
      f'bin between occupied ones:')
print('   ' + ' '.join(f'{v}' for v in hist))
interior = hist[np.argmax(hist > 0):len(hist) - np.argmax(hist[::-1] > 0)]
print(f'   occupied span: {len(interior)} bins, minimum {interior.min()} cells, '
      f'{int((interior == 0).sum())} empty')

# ---------------------------------------------------------------------------
# 8. Figures. No archetype overlay: the shared helpers close a polygon through the
#    vertices, which is meaningless across four separately-fit simplices. The
#    top-cells-only coloring shows where each archetype sits, and more honestly.
# ---------------------------------------------------------------------------
panels   = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]
panel_3d = (0, 1, 2, 'PC1', 'PC2', 'PC3')

print('\nGenerating figures...')
scatter_categorical_html(
    xp_grid=[xp],
    cell_metadata={
        'archetype':           assign['archetype'].values,
        'subclass':            assign['subclass'].values,
        'archetype_top_cells': assign['archetype_top_cells'].values,
        'archetype_by_score':  assign['archetype_by_score'].values,
        CLUSTER_COL:           types,
        SAMPLE_COL:            samples,
    },
    title='Cheng22 mouse IT — all four subclasses in one PC space, by archetype',
    out_path=OUT_HTML,
    # turbo over the natsorted labels: 11 distinct colors (the default cycle has 10) that
    # also run L2/3 -> L6IT, so a laminar gradient would read as a color gradient
    ordered_labels=('archetype', 'archetype_by_score', 'archetype_top_cells'),
    panels=panels, panel_3d=panel_3d,
)

save_archetype_scatter_pdf(
    xp, assign['archetype'].values, panels, aa=None,
    title='Cheng22 mouse IT — joint PCA, cells colored by archetype',
    out_path=OUT_PDF_ARCH, label_order=ARCH_ORDER, cmap='tab20', s=2)

save_archetype_scatter_pdf(
    xp, assign['subclass'].values, panels, aa=None,
    title='Cheng22 mouse IT — joint PCA, cells colored by subclass',
    out_path=OUT_PDF_SUB,
    label_order=[cfg['mouse_subclass'] for cfg in SUBCLASSES], cmap='tab10', s=2)

print('\nDone.')
