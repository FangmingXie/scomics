"""All human IT cells in one varimax space, colored by archetype — the human depth order.

The human analogue of script 15. Script 04 fits the 13 human archetypes (L2/3 IT A-D,
L4 IT A-C, L5 IT A-D, L6 IT A-B) one subclass at a time, so they share no coordinate
system; and 04's markers are one-vs-each *within* a subclass, which excludes by
construction any gene roughly constant across that subclass — which is what a laminar
identity gene is. Nothing computed so far can say where a human archetype sits along the
IT depth continuum, and the only human labeling in the repo is the visualization-only
L2/3 reversal that 07/10/12/13 carry (a plain A,B,C,D -> D',C',B',A' flip inherited from
l23_evo/25 to match a figure's orientation, not derived from any depth statistic).

This pools all 93,152 IT cells into one embedding, which sees the between-layer genes the
per-subclass fits discard, and colors each cell by its archetype. Three patterns are then
distinguishable by eye, as in 15:

  continuum  — one connected cloud, subclasses ordered along a single axis, archetypes
               strung along it in laminar order with boundary archetypes overlapping
  recurring  — archetype coloring runs orthogonal to the subclass axis: the same
               within-layer axis rediscovered four times
  discrete   — four separated clouds with density gaps between them

The recipe is 02.human_jorstad23_it_varimax.py:229-268 with its per-subclass loop replaced
by pooling: `.X` used directly, 2000 HVGs by variance, StandardScaler, PCA(10,
random_state=0), Kaiser varimax, components reordered by descending score variance.
Keeping the procedure identical means any difference from the per-subclass records is the
pooling, not the method.

WHY THE GEOMETRY IS READ ON VARIMAX COMPONENTS, NOT RAW PCs
On mouse, PC1/PC2 carry cell_type 0.86/0.90 with library_size 0.0006 — the depth continuum
is simply the top of the embedding, and 15 reads it there. On human it is not. A
library-size/complexity gradient is the single largest source of variance in these cells,
and in an unrotated PCA it does not occupy an axis of its own: it *mixes into* the leading
components, so PC1 is both the L4-vs-rest contrast and the depth gradient at once
(measured: PC1 cell_type 0.400, library_size 0.292, r = +0.70 with log10 nCount_RNA; and
PC1's subclass R² actually RISES from 0.463 to 0.658 once library size is regressed out —
the technical axis was masking biological signal, not creating it).

This is a pre-existing property of the dataset, not of the pooling: 02's per-subclass PC1
is library-size dominated in all four subclasses (0.570 / 0.555 / 0.089 / 0.141, with L5IT
and L6IT putting 0.407 / 0.540 on PC2). The project's answer to it is already established —
rotate to simple structure, partition the variance per component, and keep the components
the partition says are cell-type driven. That is exactly what 02 computes and what 04's
Gate-A `vx_cols` consume (they drop VX1 in every subclass). This script applies the same
gate to the joint embedding, then re-runs PCA inside the gated subspace so that a 2-D
geometry can be read there — see the ARC_SPC block for why the gate alone is not enough.

Technical factors in the partition are broadened relative to 02, which uses `nCount_RNA`
alone. Sequencing depth, detection rate and mitochondrial fraction are three faces of one
nuisance and are strongly collinear (r = 0.835 between log10 nCount_RNA and nFeature_RNA),
so partialling them separately would split their shared variance and make every component
look cleaner than it is. They are entered as ONE `technical` block of three columns:
z(log10 nCount_RNA), z(nFeature_RNA), z(mito fraction from .raw). Library size enters as
log10 because the dependence is on the log (02 z-scores the raw counts, which understates
it). Per-covariate marginal R² is printed alongside so the block stays interpretable.

Deviations from the mouse script 15, all forced by the human side:
  * Four per-subclass h5ads instead of one pooled IT file. 15's `Subclass`-values assertion
    has no analogue; the analogue is 11/13's assertion that the four `var` frames are
    identical and in the same order.
  * `.X` is already ln(CPM+1) and is used DIRECTLY. 15 renormalizes from `.raw` only
    because cheng22's `.X` is log1p(CP10k) while its `.raw` holds counts (G2). The human
    `.raw` is read here only to compute the mitochondrial fraction.
  * Gene identifiers are ENSG in `var_names`; the loadings TSVs are keyed on
    `var['feature_name']` symbols, as everywhere on the human side (G5).
  * Varimax, a Gate-A cut on the variance partition, and a second PCA *inside* the gated
    subspace — the geometry is read on SPC1-SPC2, not on PC1-PC2. Mouse 15 needs two axes
    to carry the cell-type structure and on mouse PC1-PC2 already do; here it takes all
    three steps to get there. Figures are emitted for all three bases — each on its own
    most subclass-informative gated components, so the comparison is like for like — and the
    basis-comparison table quantifies what the panels show.
  * The variance partition takes four factors (cell_type / donor / source / technical),
    not 15's three — `development_stage` stays excluded, being 1:1 with `donor_id`.
  * No interactive HTML: at 93,152 cells the per-category Plotly traces are not worth the
    file. Only the two vector PDFs are written.

THE ARC ORDER IS THE DELIVERABLE, AND IT IS PRODUCED IN TWO RUNS — see ARC_ORDER_CURATED
below. Run 1 writes the raw angular sort with `curated=False` in every row of
OUT_ARC_ORDER; that file is a PROPOSAL and any future consumer must refuse it, e.g.
`if not arc_df['curated'].all(): raise ...`. Only after the run-1 diagnostics and figures
have been read does ARC_ORDER_CURATED get filled in and the file become a depth record.

Caveats:
  * Pooled HVGs are dominated by between-subclass genes. That is the point here, but it
    makes this a new space, not comparable to the per-subclass varimax records, and it must
    not be used to re-derive archetypes.
  * Cell counts are unbalanced far worse than mouse (L2/3 IT 47125 / L4 IT 30455 /
    L5 IT 10537 / L6 IT 5035, i.e. 50.6/32.7/11.3/5.4 %), so the components are weighted
    toward L2/3 and L4. The centroids themselves are each the mean of exactly N_TOP_CELLS
    cells, equal by construction, so class size enters only through the HVG set and the
    component directions; the top-cell vs all-cell centroid order comparison tests that.
  * A 2-D plane is a slice: subclasses that separate cleanly in the full gated subspace can
    still overlap in any one plane of it. The plane scan reports kNN subclass purity per
    plane against the full-subspace ceiling so that gap is visible rather than assumed.
  * This script's tables keep 04's A/B/C/D letters — it is the derivation of record. The
    primed labels it proposes live only in OUT_ARC_ORDER's `new_label`.
  * PCHA's convex weights S would be the canonical archetype assignment, but 04 does not
    save them. Both proxies are emitted — nearest vertex in that subclass's own PCHA space,
    and argmax of 04's scores — plus a top-cells-only label.
  * If `donor` survives the gate on the components the arc is read in, the follow-up is a
    Harmony-corrected variant; precedent in scripts/it/28-39.harmony.*.

Reads:
  links/it_evo/jorstad23_human_WithinArea_{L23IT,L4IT,L5IT,L6IT}.h5ad
  local_data/res/it_evo/04.human_<TOKEN>_pcha_{xp,aa}.tsv
  local_data/res/it_evo/04.human_<TOKEN>_archetype_scores.tsv
  local_data/res/it_evo/04.human_<TOKEN>_top_cells.tsv
Outputs:
  local_data/res/it_evo/22.human_IT_joint_pca_{coords,loadings}.tsv
  local_data/res/it_evo/22.human_IT_joint_varimax_{coords,loadings}.tsv
  local_data/res/it_evo/22.human_IT_joint_spc_coords.tsv
  local_data/res/it_evo/22.human_IT_joint_spc_loadings_{on_vx,gene}.tsv
  local_data/res/it_evo/22.human_IT_joint_{pc,vx,spc}_variance_partition.tsv
  local_data/res/it_evo/22.human_IT_joint_plane_scan.tsv   (naive-PC, gated-VX and
      SPC planes, each with kNN subclass purity and subclass contiguity)
  local_data/res/it_evo/22.human_IT_joint_component_informativeness.tsv  (per component
      of all three bases: cell_type partial R², and marginal subclass / archetype R²)
  local_data/res/it_evo/22.human_IT_joint_basis_comparison.tsv
  local_data/res/it_evo/22.human_IT_joint_archetype_assignment.tsv
  local_data/res/it_evo/22.human_IT_joint_archetype_centroids.tsv
  local_data/res/it_evo/22.human_IT_joint_archetype_arc_order.tsv   (the depth order and the
      A'/B'/... display labels, carrying both the curated rank and this script's angular
      rank, plus the `curated` flag that says whether the two differ by inspection or not)
  local_data/fig/it_evo/22.human_IT_joint_{pc,vx,spc}_{archetype,subclass}.pdf
      (three bases x two colorings; the geometry below is read on the SPC basis)
"""

import os
import sys
import gc
import itertools
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from viz import save_archetype_scatter_pdf

# --- file paths ---
LINK_DIR      = os.path.join(PROJECT_ROOT, 'links', 'it_evo')
OUT_RES_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
OUT_FIG_DIR   = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_PC_COORDS = os.path.join(OUT_RES_DIR, '22.human_IT_joint_pca_coords.tsv')
OUT_PC_LOAD   = os.path.join(OUT_RES_DIR, '22.human_IT_joint_pca_loadings.tsv')
OUT_VX_COORDS = os.path.join(OUT_RES_DIR, '22.human_IT_joint_varimax_coords.tsv')
OUT_VX_LOAD   = os.path.join(OUT_RES_DIR, '22.human_IT_joint_varimax_loadings.tsv')
OUT_PC_VAR    = os.path.join(OUT_RES_DIR, '22.human_IT_joint_pc_variance_partition.tsv')
OUT_VX_VAR    = os.path.join(OUT_RES_DIR, '22.human_IT_joint_vx_variance_partition.tsv')
OUT_SCAN      = os.path.join(OUT_RES_DIR, '22.human_IT_joint_plane_scan.tsv')
OUT_INFORM    = os.path.join(OUT_RES_DIR,
                             '22.human_IT_joint_component_informativeness.tsv')
OUT_BASIS     = os.path.join(OUT_RES_DIR, '22.human_IT_joint_basis_comparison.tsv')
OUT_ASSIGN    = os.path.join(OUT_RES_DIR, '22.human_IT_joint_archetype_assignment.tsv')
OUT_CENTROIDS = os.path.join(OUT_RES_DIR, '22.human_IT_joint_archetype_centroids.tsv')
OUT_ARC_ORDER = os.path.join(OUT_RES_DIR, '22.human_IT_joint_archetype_arc_order.tsv')
OUT_SPC_COORDS = os.path.join(OUT_RES_DIR, '22.human_IT_joint_spc_coords.tsv')
OUT_SPC_LOAD_VX = os.path.join(OUT_RES_DIR, '22.human_IT_joint_spc_loadings_on_vx.tsv')
OUT_SPC_LOAD_GN = os.path.join(OUT_RES_DIR, '22.human_IT_joint_spc_loadings_gene.tsv')
OUT_SPC_VAR   = os.path.join(OUT_RES_DIR, '22.human_IT_joint_spc_variance_partition.tsv')
# One pair of PDFs per basis: the naive PCs, the varimax components, and the PCs of the
# gated varimax subspace. Reading them left to right is the argument of this script.
OUT_PDF = {b: {c: os.path.join(OUT_FIG_DIR, f'22.human_IT_joint_{b}_{c}.pdf')
               for c in ('archetype', 'subclass')}
           for b in ('pc', 'vx', 'spc')}

# `noc` must match script 04's noc for the same token. The viz-only `rename`/`flip` that 13
# carries for L23 are deliberately NOT propagated here: this is the derivation of record and
# its tables are keyed by 04's A/B/C/D letters.
SUBCLASSES = [
    {'token': 'L23',  'human_subclass': 'L2/3 IT',
     'h5ad': 'jorstad23_human_WithinArea_L23IT.h5ad', 'noc': 4},
    {'token': 'L4',   'human_subclass': 'L4 IT',
     'h5ad': 'jorstad23_human_WithinArea_L4IT.h5ad',  'noc': 3},
    {'token': 'L5IT', 'human_subclass': 'L5 IT',
     'h5ad': 'jorstad23_human_WithinArea_L5IT.h5ad',  'noc': 4},
    {'token': 'L6IT', 'human_subclass': 'L6 IT',
     'h5ad': 'jorstad23_human_WithinArea_L6IT.h5ad',  'noc': 2},
]

# --- parameters (N_HVG / N_PCS / column names from 02) ---
SUBCLASS_COL      = 'WithinArea_subclass'
CLUSTER_COL       = 'WithinArea_cluster'
DONOR_COL         = 'donor_id'
SOURCE_COL        = 'Source'
LIBSIZE_COL       = 'nCount_RNA'
NGENE_COL         = 'nFeature_RNA'
GENE_NAME_COL     = 'feature_name'
MITO_PREFIX       = 'MT-'
N_HVG             = 2000
N_PCS             = 10
PCA_RANDOM_STATE  = 0
ALPHABET          = ['A', 'B', 'C', 'D', 'E', 'F']
N_CELLS_EXPECTED  = {'L23': 47125, 'L4': 30455, 'L5IT': 10537, 'L6IT': 5035}
N_CLUSTERS_POOLED = 24          # 6 + 8 + 8 + 2
N_DONORS          = 3
N_SOURCES         = 2
ARC_BINS          = 36          # 10-degree bins around the arc, for the connectedness check
INTERIOR_FRAC     = 0.5         # centroid radius below this x median -> its angle is unreliable
KNN_K             = 25          # neighbourhood size for the subclass-purity diagnostic
KNN_N            = 20000        # seeded subsample it is measured on
KNN_SEED          = 0
# The two subclasses that fix the arc's READING DIRECTION (see arc_from_centroids).
DIR_FIRST, DIR_LAST = 'L2/3 IT', 'L6 IT'

# ===========================================================================
# JOINT-EMBEDDING GATE A, then the plane the geometry is read in.
#
# Gate A keeps the VX components the variance partition calls cell-type driven (cell_type
# strictly greater than donor, source AND technical). Varimax puts each nuisance on its own
# axis — measured here: library size and detection rate on VX1, mitochondrial fraction on
# VX2, donor on VX5 — so the gate is a clean cut rather than a judgement call.
#
# But varimax also SPREADS the cell-type signal across all seven surviving components (0.78
# down to 0.40), and no single plane of them holds it: of the 21 gated planes, the best
# reaches kNN subclass purity 0.886 against a 0.951 seven-component ceiling, and NOT ONE
# keeps all four subclasses contiguous around the arc. A 2-D arc cannot be read there.
#
# Re-running PCA *inside* the gated subspace fixes that. It is only a rotation of that
# subspace, so it cannot recover anything the gate discarded — it redistributes what is
# there, concentrating 61% of the subspace variance and 0.939 of the 0.946 purity into two
# axes. That restores the condition mouse 15 relies on (two axes carrying the cell-type
# structure) without importing any new assumption.
#
# ARC_SPC is therefore simply the top two by variance. No selection is applied and, in
# particular, anatomical depth order plays no part in it — so the subclass sequence the arc
# comes out in stays independent evidence about the data. The scan below reports every SPC
# plane's purity so that claim is checkable rather than asserted.
#
# Two independent rankings agree on it. The informativeness table ranks components by
# marginal between-subclass R², and its top two are SPC2 (0.773) and SPC1 (0.688) — the same
# pair variance picks. That table is also what explains an earlier wrong turn: selecting on
# `cell_type` partial R² instead picks VX3-VX4, because partial R² asks only whether an axis
# is uncontaminated, not whether it carries the between-subclass contrast. VX4 is clean
# (cell_type 0.773) but almost entirely within-subclass (subclass R² 0.363), while VX6 is
# less clean (0.690) and far more informative (0.758). Ranking the gated VX by subclass R²
# gives VX6, VX3 — exactly the highest-purity VX plane. Both rankings are emitted so the
# distinction stays visible.
# ===========================================================================
ARC_SPC = ('SPC1', 'SPC2')
# ===========================================================================

# ===========================================================================
# THE DELIVERABLE. Unlike mouse 15, the human depth order is not known in advance — it is
# what this script is for. Two runs:
#   RUN 1  ARC_ORDER_CURATED is None. The script prints the angular order and the geometry
#          diagnostics, writes both PDFs, and writes OUT_ARC_ORDER with arc_rank ==
#          arc_rank_angular and curated=False in every row. Nothing downstream may read a
#          curated=False file as a depth record.
#   RUN 2  Inspect the run-1 figures and diagnostics, list the 13 labels below in depth
#          order, rerun. arc_rank becomes the curated rank, arc_rank_angular is unchanged
#          (the geometry is deterministic), curated=True, and the disagreement between the
#          two is reported as inverted pairs.
# ===========================================================================
# Run 2. This is the run-1 angular order with ONE segment overruled: L5 IT is taken in the
# order D, C, B, A rather than the angular D, B, C, A. The three overruled centroids span
# 12.3 degrees of angle (L5 B -91.9, L5 C -80.8, L5 A -79.6) and sit 0.98-2.92 units apart —
# L5 A/C are the closest pair of all 78 — so the angular sort has no resolving power there,
# exactly the situation that forced the mouse L6IT correction in 15. Curated by inspection.
ARC_ORDER_CURATED = [
    'L2/3 IT D', 'L2/3 IT C', 'L2/3 IT B', 'L2/3 IT A',
    'L4 IT C',   'L4 IT B',   'L4 IT A',
    'L5 IT D',   'L5 IT C',   'L5 IT B',   'L5 IT A',
    'L6 IT A',   'L6 IT B',
]
# ===========================================================================

UNCURATED_BANNER = (
    '\n' + '*' * 78 +
    '\n*** ARC_ORDER_CURATED IS None — THIS IS RUN 1 (UNCURATED).\n'
    '*** The arc order written below is the raw angular sort. It is a PROPOSAL, not the\n'
    '*** depth record. Inspect the two PDFs and the geometry diagnostics, then set\n'
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


def build_factors(cluster, donor, source, tech):
    """02's build_factors on POOLED plain-string arrays, with a grouped technical block.

    Strings, not Categoricals: `pd.concat` of Categoricals whose level sets differ per file
    is dtype-version-dependent, and a surviving unused level becomes an all-zero dummy
    column (G6). `pd.get_dummies` on an object array emits one column per *observed* value
    by construction, so the hazard is removed at the source; 02's guard is kept as the
    invariant check.

    `tech` is a (n_cells, 3) block — log10 library size, genes detected, mito fraction —
    entered as ONE factor because the three are collinear (r = 0.835 for the first two) and
    partialling them separately would attribute their shared variance to neither.
    """
    for name, arr in (('cell_type', cluster), ('donor', donor), ('source', source)):
        if getattr(arr, 'dtype', None) == 'category':
            raise ValueError(f'{name} was passed as a Categorical — pool it as str (G6)')
    factors = {
        'cell_type': pd.get_dummies(cluster).values.astype(float),
        'donor':     pd.get_dummies(donor).values.astype(float),
        'source':    pd.get_dummies(source).values.astype(float),
        'technical': np.apply_along_axis(zscore, 0, tech),
    }
    expected = {'cell_type': N_CLUSTERS_POOLED, 'donor': N_DONORS, 'source': N_SOURCES}
    for name, n_exp in expected.items():
        col_sums = factors[name].sum(axis=0)
        if (col_sums == 0).any():
            raise ValueError(f'{name} design matrix has {(col_sums == 0).sum()} all-zero '
                             f'dummy column(s) — stale categorical levels survived (G6)')
        if factors[name].shape[1] != n_exp:
            raise ValueError(f'pooled {name} has {factors[name].shape[1]} levels, expected '
                             f'{n_exp} — the inputs are not the four IT subsets this script '
                             f'was written against')
    return factors


def variance_partition(scores, comp_cols, factors, tech, tech_names, label):
    """Partial R² per component for each factor in `factors` (02's dict signature).

    `tech` / `tech_names` are only for the printed per-covariate marginal R², which keeps the
    grouped `technical` block interpretable — they take no part in the partition itself.
    """
    X_full = np.hstack(list(factors.values()))

    print(f'\nVariance partitioning ({label}):')
    rows = []
    for i, col in enumerate(comp_cols):
        y = scores[:, i]
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


def arc_from_centroids(P):
    """Order the centroids around the arc in the P (2-column) plane.

    A 1-D gradient renders as a curve in 2-D (horseshoe), so the ordering along it — not the
    2-D spread — is what carries the continuum claim. Two refinements over 15, which could
    hardcode its answer and here cannot:

      * the circle is cut at the LARGEST ANGULAR GAP, not at the -180 deg wraparound, so an
        arc that straddles the wraparound is not split in the middle;
      * the traversal is oriented so DIR_FIRST precedes DIR_LAST. A component has an
        arbitrary sign, which would otherwise reverse the printed order run to run. This
        fixes a READING DIRECTION only — the order up to reversal is untouched — it is not
        a laminar prior on the content.
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


def subclass_runs(arc):
    """The subclass sequence along `arc`, collapsing only CONSECUTIVE repeats.

    `drop_duplicates` would collapse non-consecutive ones too, so an arc that leaves a
    subclass and comes back to it (L2/3, L4, L6, L5, L4) would be reported as four tidy
    contiguous blocks. Run-length encoding is what the contiguity claim actually needs.
    """
    return [k for k, _ in itertools.groupby(subclass_of(n) for n in arc)]


def cyclic_offset(order_a, order_b):
    """Is `order_b` the same cycle as `order_a`? Returns (rotation, reversed) or None.

    The centroids form a closed-ish loop, so an ordering is only defined up to where the
    circle was cut and which way it was traversed. Comparing two such orders with plain
    `inverted_pairs` charges the full cost of a rotation — the size-weighted control below
    scored 36 of 78 pairs against an order it in fact matches exactly — so a cyclic
    comparison is what the geometry actually calls for.
    """
    n = len(order_a)
    for rev in (False, True):
        b = order_b[::-1] if rev else list(order_b)
        for k in range(n):
            if b[k:] + b[:k] == list(order_a):
                return k, rev
    return None


def inverted_pairs(order_a, order_b):
    """Pairs ordered one way in `order_a` and the other way in `order_b`.

    Pairs, not rank shifts: moving one archetype past others renumbers them without
    reordering them relative to each other, and calling that a disagreement would overstate
    how much of one order the other overruled.
    """
    rank_b = {n: i for i, n in enumerate(order_b)}
    return [(a, b) for a, b in itertools.combinations(order_a, 2) if rank_b[a] > rank_b[b]]


# ---------------------------------------------------------------------------
# 1. Pool the four per-subclass h5ads. `.X` stays sparse; only per-gene moments are
#    accumulated here, so the 93,152 x 29,352 dense matrix (10.9 GB) is never built.
# ---------------------------------------------------------------------------
gene_ensg_ref, gene_symbol_ref, mito_mask = None, None, None
sum_x = sum_x2 = None
X_blocks, barcodes_by_token = [], {}
meta = {c: [] for c in (CLUSTER_COL, DONOR_COL, SOURCE_COL)}
tech_parts, barcode_parts, subclass_parts = [], [], []

for cfg in SUBCLASSES:
    token, subclass = cfg['token'], cfg['human_subclass']
    in_h5ad = os.path.join(LINK_DIR, cfg['h5ad'])
    print(f'\nLoading {in_h5ad}...')
    adata = ad.read_h5ad(in_h5ad)
    print(f'  {adata.n_obs} cells x {adata.n_vars} genes')

    if adata.n_obs != N_CELLS_EXPECTED[token]:
        raise ValueError(f'{token}: {adata.n_obs} cells but {N_CELLS_EXPECTED[token]} '
                         f'expected — this is not the subset scripts 02/04 were fit on')
    present = set(pd.unique(adata.obs[SUBCLASS_COL].astype(str)))
    if present != {subclass}:
        raise ValueError(f'{token}: {SUBCLASS_COL} values {sorted(present)} != '
                         f'{{{subclass!r}}} — each h5ad must hold exactly its own subclass')

    gene_ensg = adata.var_names.values
    gene_symbol = adata.var[GENE_NAME_COL].astype(str).values
    if gene_ensg_ref is None:
        if len(set(gene_symbol)) != len(gene_symbol):
            raise ValueError(f'{GENE_NAME_COL} is not unique in {cfg["h5ad"]} — the pooled '
                             f'loadings index assumes one row per gene symbol')
        gene_ensg_ref, gene_symbol_ref = gene_ensg, gene_symbol
        mito_mask = np.array([g.startswith(MITO_PREFIX) for g in gene_symbol])
        if not mito_mask.any():
            raise ValueError(f'no {MITO_PREFIX}* genes in {GENE_NAME_COL} — the mitochondrial '
                             f'fraction cannot be computed and the technical block would be '
                             f'silently one covariate short')
        print(f'  {int(mito_mask.sum())} mitochondrial genes for the technical block')
        sum_x  = np.zeros(adata.n_vars, dtype=np.float64)
        sum_x2 = np.zeros(adata.n_vars, dtype=np.float64)
    elif not (np.array_equal(gene_ensg, gene_ensg_ref)
              and np.array_equal(gene_symbol, gene_symbol_ref)):
        raise ValueError(f'{cfg["h5ad"]} var differs from the first h5ad — the joint '
                         f'embedding requires all four subclasses to share one gene index '
                         f'in one order')

    X = adata.X
    if not sp.isspmatrix_csr(X):
        raise ValueError(f'{token}: .X is {type(X).__name__}, expected csr_matrix — the '
                         f'moment accumulation below indexes CSR column indices')
    X.sum_duplicates()          # canonical form; np.bincount over .indices assumes it
    # float64 moments. scipy's X.sum(axis=0) accumulates in float32 and was measured off by
    # 0.11 absolute on these matrices — far too coarse to rank genes by variance.
    sum_x  += np.bincount(X.indices, weights=X.data.astype(np.float64), minlength=X.shape[1])
    sum_x2 += np.bincount(X.indices, weights=X.data.astype(np.float64) ** 2,
                          minlength=X.shape[1])
    X_blocks.append(X)

    # mitochondrial fraction from the raw counts (the only use of .raw on the human side)
    if adata.raw is None or adata.raw.shape != adata.shape:
        raise ValueError(f'{token}: .raw is missing or not aligned with .X — the mito '
                         f'fraction is computed on raw counts')
    raw_tot  = np.asarray(adata.raw.X.sum(axis=1)).ravel()
    raw_mito = np.asarray(adata.raw.X[:, mito_mask].sum(axis=1)).ravel()
    if (raw_tot <= 0).any():
        raise ValueError(f'{token}: {(raw_tot <= 0).sum()} cells have zero raw counts')
    tech_parts.append(np.column_stack([
        np.log10(adata.obs[LIBSIZE_COL].values.astype(float)),
        adata.obs[NGENE_COL].values.astype(float),
        raw_mito / raw_tot,
    ]))

    for c in meta:
        meta[c].append(adata.obs[c].astype(str).values)     # str, never Categorical (G6)
    barcodes_by_token[token] = adata.obs_names.values
    barcode_parts.append(adata.obs_names.values)
    subclass_parts.append(np.full(adata.n_obs, subclass))

    del adata
    gc.collect()

barcodes   = np.concatenate(barcode_parts)
subclasses = np.concatenate(subclass_parts)
tech       = np.vstack(tech_parts)
meta       = {c: np.concatenate(v) for c, v in meta.items()}
n_pooled   = len(barcodes)
TECH_NAMES = ['log10_libsize', 'n_genes', 'mito_frac']

if len(set(barcodes)) != n_pooled:
    raise ValueError(f'{n_pooled - len(set(barcodes))} duplicate barcodes across the four '
                     f'h5ads — the pooled cell index must be unique')
print(f'\nPooled: {n_pooled} cells x {len(gene_ensg_ref)} genes')
print('  cells per subclass: '
      + str(pd.Series(subclasses).value_counts().reindex(
          [c['human_subclass'] for c in SUBCLASSES]).to_dict()))
print('  technical covariates (median): '
      + ', '.join(f'{n}={np.median(tech[:, j]):.3f}' for j, n in enumerate(TECH_NAMES))
      + f'   r(log10_libsize, n_genes)={np.corrcoef(tech[:, 0], tech[:, 1])[0, 1]:.3f}')

# --- 2. HVG selection on the pooled cells (02:236-241, computed sparsely) ---
print(f'\nSelecting top {N_HVG} HVGs across all human IT cells...')
gene_var = sum_x2 / n_pooled - (sum_x / n_pooled) ** 2
order    = np.argsort(gene_var)[::-1]
gap      = gene_var[order[N_HVG - 1]] - gene_var[order[N_HVG]]
print(f'  variance at the cutoff: rank {N_HVG - 1}={gene_var[order[N_HVG - 1]]:.6f}, '
      f'rank {N_HVG}={gene_var[order[N_HVG]]:.6f}, gap={gap:.3e}')
if gap <= 0:
    raise ValueError(f'exact variance tie at the {N_HVG}-gene cutoff — the HVG set would be '
                     f'argsort-order-dependent, not a property of the data')
hvg_idx   = order[:N_HVG]
hvg_names = gene_symbol_ref[hvg_idx]        # G5: symbols, not ENSG

# G7: slice the sparse columns, then densify. 93,152 x 2000 float32 = 745 MB.
X_hvg = np.vstack([Xb[:, hvg_idx].toarray() for Xb in X_blocks])
del X_blocks
gc.collect()

# The sparse moments must reproduce a plain densified variance. Compared in float64: against
# X_hvg.var(axis=0) (float32) this would need atol~1e-3 and would prove nothing.
var_dense = X_hvg.astype(np.float64).var(axis=0)
if not np.allclose(gene_var[hvg_idx], var_dense, rtol=1e-9, atol=0):
    k = int(np.argmax(np.abs(gene_var[hvg_idx] - var_dense)))
    raise ValueError(f'sparse moment variance disagrees with the densified block at '
                     f'{hvg_names[k]}: {gene_var[hvg_idx][k]:.9f} vs {var_dense[k]:.9f}')
del var_dense
gc.collect()

# --- 3. Scale + PCA + varimax (02:246-275) ---
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
# 4. Archetype assignment per cell, from the per-subclass PCHA records of script 04
# ---------------------------------------------------------------------------
print('\nAssigning cells to archetypes...')
blocks = []
for cfg in SUBCLASSES:
    token, subclass, noc = cfg['token'], cfg['human_subclass'], cfg['noc']
    xp_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_xp.tsv'),
                            sep='\t', index_col=0)
    aa_df     = pd.read_csv(os.path.join(OUT_RES_DIR, f'04.human_{token}_pcha_aa.tsv'),
                            sep='\t', index_col=0)
    scores_df = pd.read_csv(os.path.join(OUT_RES_DIR,
                                         f'04.human_{token}_archetype_scores.tsv'),
                            sep='\t', index_col=0)
    top_df    = pd.read_csv(os.path.join(OUT_RES_DIR, f'04.human_{token}_top_cells.tsv'),
                            sep='\t')

    if len(aa_df) != noc:
        raise ValueError(f'{token}: pcha_aa has {len(aa_df)} archetypes but noc={noc}')
    if list(aa_df.columns) != list(xp_df.columns):
        raise ValueError(f'{token}: pcha_aa columns {list(aa_df.columns)} != pcha_xp columns '
                         f'{list(xp_df.columns)} — the vertices are not in the cells\' space')
    # The h5ad obs order and 04's cell order are the same construction, so this is an
    # assertion rather than a reindex: a mismatch means one of them was regenerated.
    if not np.array_equal(barcodes_by_token[token], xp_df.index.values):
        raise ValueError(f'{token}: the h5ad cell order differs from '
                         f'04.human_{token}_pcha_xp.tsv — one of them is stale')
    if not scores_df.index.equals(xp_df.index):
        raise ValueError(f'{token}: 04 scores and 04 pcha_xp disagree on cell order')
    if not set(top_df['cell']).issubset(set(xp_df.index)):
        raise ValueError(f'{token}: 04 top_cells names cells absent from 04 pcha_xp')

    names = [f'{subclass} {ALPHABET[k]}' for k in range(noc)]

    # nearest archetype vertex in this subclass's own PCHA space (all PC columns)
    d = np.stack([np.linalg.norm(xp_df.values - aa_df.values[k], axis=1) for k in range(noc)],
                 axis=1)
    nearest = np.array(names)[d.argmin(axis=1)]

    # argmax of 04's own [0,1] scores — an independent proxy for the same thing
    score_cols = [f'score_{ALPHABET[k]}' for k in range(noc)]
    by_score = np.array(names)[scores_df[score_cols].values.argmax(axis=1)]

    # top cells only: where each archetype actually is, everything else greyed out
    top_label = pd.Series('other', index=xp_df.index)
    for k in range(noc):
        cells = top_df[top_df['archetype'] == f'archetype_{k + 1}']['cell'].values
        top_label.loc[cells] = names[k]

    blocks.append(pd.DataFrame({'human_subclass': subclass,
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
if not np.array_equal(assign.index.values, barcodes):
    raise ValueError('the assignment cell order differs from the embedding cell order — '
                     'both are built by looping SUBCLASSES in order, so this cannot drift '
                     'without one of the inputs having been regenerated')
assign[CLUSTER_COL] = meta[CLUSTER_COL]
assign[DONOR_COL]   = meta[DONOR_COL]
assign.to_csv(OUT_ASSIGN, sep='\t')
print(f'Saved {OUT_ASSIGN}')

ARCH_ORDER = [f'{cfg["human_subclass"]} {ALPHABET[k]}'
              for cfg in SUBCLASSES for k in range(cfg['noc'])]

# ---------------------------------------------------------------------------
# 5. What do the components encode? This is what decides which are interpretable.
# ---------------------------------------------------------------------------
factors = build_factors(meta[CLUSTER_COL], meta[DONOR_COL], meta[SOURCE_COL], tech)
pc_var  = variance_partition(scores, pc_cols, factors, tech, TECH_NAMES,
                             'joint human IT PCs — pre-varimax')
pc_var.to_csv(OUT_PC_VAR, sep='\t')
vx_var  = variance_partition(vx_scores, vx_cols, factors, tech, TECH_NAMES,
                             'joint human IT VX — post-varimax')
vx_var.to_csv(OUT_VX_VAR, sep='\t')
print(f'Saved {OUT_PC_VAR} and {OUT_VX_VAR}')

TECH_FACTORS = ('donor', 'source', 'technical')


def gate_of(var_df, cols, label):
    """Components whose cell_type partial R² beats every technical factor's.

    Applied to all three bases. On VX it is Gate A proper — the SPC basis is built from what
    it keeps. On the naive PCs it selects nothing (the PC basis is not used for geometry);
    it is reported so the three bases are judged by one criterion instead of the PC row
    being exempt from the standard the others are held to.
    """
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
if len(gate_a) < 2:
    raise ValueError(f'Gate A kept {len(gate_a)} VX components ({gate_a}) — at least two are '
                     f'needed to read a 2-D arc. The joint embedding is technical-dominated; '
                     f'the follow-up is a Harmony/regression-corrected variant '
                     f'(scripts/it/28-39.harmony.*)')

vx_df_all = pd.DataFrame(vx_scores, index=barcodes, columns=vx_cols)
if len(gate_a) < 3:
    raise ValueError(f'Gate A kept only {len(gate_a)} components ({gate_a}); the subspace PCA '
                     f'and its three-panel figures need at least 3')

# ---------------------------------------------------------------------------
# 6. PCA inside the gated subspace. A rotation of it — no new information, but it puts the
#    cell-type structure back into leading axes so a 2-D geometry can be read honestly.
# ---------------------------------------------------------------------------
gate_idx  = [vx_cols.index(c) for c in gate_a]
spc_pca   = PCA(n_components=len(gate_a), random_state=PCA_RANDOM_STATE)
spc_scores = spc_pca.fit_transform(vx_scores[:, gate_idx])
spc_cols  = [f'SPC{i + 1}' for i in range(len(gate_a))]
spc_df_all = pd.DataFrame(spc_scores, index=barcodes, columns=spc_cols)
print(f'\nPCA within the {len(gate_a)}-component Gate-A subspace -> {spc_cols}')
print('  explained variance ratio: '
      + ', '.join(f'{c}={v:.3f}' for c, v in zip(spc_cols, spc_pca.explained_variance_ratio_))
      + f'   (top two {spc_pca.explained_variance_ratio_[:2].sum():.3f})')

spc_df_all.to_csv(OUT_SPC_COORDS, sep='\t')
pd.DataFrame(spc_pca.components_.T, index=gate_a, columns=spc_cols).to_csv(OUT_SPC_LOAD_VX,
                                                                          sep='\t')
# composed back to genes, so an SPC can be read as a gene program (G5 symbols)
pd.DataFrame(vx_loadings[:, gate_idx] @ spc_pca.components_.T,
             index=hvg_names, columns=spc_cols).to_csv(OUT_SPC_LOAD_GN, sep='\t')
print(f'Saved {OUT_SPC_COORDS}, {OUT_SPC_LOAD_VX} and {OUT_SPC_LOAD_GN}')

spc_var = variance_partition(spc_scores, spc_cols, factors, tech, TECH_NAMES,
                             'joint human IT SPC — PCs of the gated subspace')
spc_var.to_csv(OUT_SPC_VAR, sep='\t')
print(f'Saved {OUT_SPC_VAR}')
leaked = [c for c in spc_cols
          if any(spc_var.loc[c, 'cell_type'] <= spc_var.loc[c, t] for t in TECH_FACTORS)]
if leaked:
    raise ValueError(f'{leaked} are not cell-type dominated, but every SPC is a combination '
                     f'of Gate-A components only — a technical factor cannot enter here '
                     f'unless the gate or the partition is inconsistent')

# ---------------------------------------------------------------------------
# 7. Which components actually carry cell identity, in all three bases
# ---------------------------------------------------------------------------
X_SUBCLASS  = pd.get_dummies(subclasses).values.astype(float)
X_ARCHETYPE = pd.get_dummies(assign['archetype'].values).values.astype(float)


def informativeness(scores_arr, cols, var_df, gate, basis):
    """How much of each component is cell identity, at the two levels that matter here.

    `cell_type` in the partition is the 24 WithinArea_cluster levels partialled against the
    technical factors, which answers "is this axis contaminated". It does NOT answer "does
    this axis carry the between-subclass contrast the arc is read from" — a component can be
    cleanly cell-type driven and still describe only within-subclass variation (VX8 and SPC5
    are exactly that). `subclass_r2` and `archetype_r2` are plain marginal R² on the four
    subclasses and the 13 archetypes, and they are the ranking the basis comparison below is
    actually about.
    """
    rows = [{'basis': basis, 'component': c,
             'cell_type_partial': round(var_df.loc[c, 'cell_type'], 3),
             'subclass_r2': round(r2(X_SUBCLASS, scores_arr[:, i]), 3),
             'archetype_r2': round(r2(X_ARCHETYPE, scores_arr[:, i]), 3),
             'donor': round(var_df.loc[c, 'donor'], 3),
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
# 8. Centroids, and the diagnostics that say whether a 2-D read is legitimate
# ---------------------------------------------------------------------------
centroids = pd.DataFrame(
    [spc_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
centroids.to_csv(OUT_CENTROIDS, sep='\t')
print(f'Saved {OUT_CENTROIDS}')

# kNN subclass purity: how much of the gated subspace's separation survives in one plane.
# A 2-D slice can mix subclasses the full subspace separates cleanly, and the arc is read in
# a slice, so the gap between the two is measured rather than assumed.
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
    print(f'\n{label} — kNN(k={KNN_K}) subclass purity on a seeded {len(knn_idx)}-cell '
          f'subsample; chance={knn_null:.3f}, full {len(cols)}-component subspace={ceiling:.3f}:')
    rows = []
    for x, y in itertools.combinations(cols, 2):
        arc_xy, _, _, gap_xy = arc_from_centroids(cent_frame[[x, y]])
        runs_xy = subclass_runs(arc_xy)
        rows.append({'plane': f'{x}-{y}', 'x': x, 'y': y,
                     'knn_purity': round(knn_purity(frame, (x, y)), 3),
                     'subclass_seq': ' '.join(t.replace(' IT', '') for t in runs_xy),
                     'n_runs': len(runs_xy),
                     'contiguous': len(runs_xy) == len(SUBCLASSES),
                     'depth_order': runs_xy == [c['human_subclass'] for c in SUBCLASSES]})
    df = pd.DataFrame(rows).sort_values('knn_purity', ascending=False)
    print(f'  {int(df["contiguous"].sum())} of {len(df)} planes keep every subclass '
          f'contiguous; {int(df["depth_order"].sum())} also give the anatomical order')
    print(df.drop(columns=['x', 'y']).to_string(index=False))
    return df


vx_ceiling  = knn_purity(vx_df_all, gate_a)
spc_ceiling = knn_purity(spc_df_all, spc_cols)
pc_df_all    = pd.DataFrame(scores, index=barcodes, columns=pc_cols)
centroids_pc = pd.DataFrame(
    [pc_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
centroids_vx = pd.DataFrame(
    [vx_df_all[assign['archetype_top_cells'] == name].mean() for name in ARCH_ORDER],
    index=ARCH_ORDER)
pc_ceiling = knn_purity(pc_df_all, gate_pc)
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

# The three-line summary the whole basis argument rests on. Emitted rather than asserted:
# a 2-D read is only legitimate where the best plane approaches its own subspace ceiling
# AND some plane keeps the subclasses contiguous.
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
print(f'\nBasis comparison — chance purity {knn_null:.3f}:')
print(basis_df.to_string(index=False))
print(f'Saved {OUT_BASIS}')

bad = [c for c in ARC_SPC if c not in spc_cols]
if bad:
    raise ValueError(f'ARC_SPC names {bad}, which are not among {spc_cols}')
print(f'\nGeometry is read on {ARC_SPC} — the top two subspace PCs by variance, '
      f'no selection applied.')
print(f'  purity in this plane {knn_purity(spc_df_all, ARC_SPC):.3f} vs {spc_ceiling:.3f} for '
      f'the full {len(spc_cols)}-component subspace '
      f'(best gated VX plane managed {scan_vx["knn_purity"].max():.3f})')

# ---------------------------------------------------------------------------
# 9. Figures — one pair per basis, each basis shown on ITS OWN most subclass-informative
#    gated components (panel 1 is therefore that basis's best plane from the scan above).
#
#    Not the first three components of each basis: in the PC and VX bases those are the
#    nuisance axes the gate rejects (PC1/PC2 technical and donor; VX1 library size, VX2
#    mitochondrial), so plotting them would show PC and VX at their worst while showing SPC
#    at its best — the comparison would be rigged by the panel choice rather than by the
#    bases. Ranking by marginal subclass R² and taking the top three gated components asks
#    the same question of all three. What the naive PCs lose is then a property of the basis,
#    not of the panels: PC1 is excluded because it is contaminated, and that is the finding.
#
#    No archetype overlay: the shared helper closes a polygon through the vertices, which is
#    meaningless across four separately-fit simplices.
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
    top3 = list(ranked['component'])[:3]
    r2s  = {c: ranked.set_index('component').loc[c, 'subclass_r2'] for c in top3}
    idx  = [cols.index(c) for c in top3]
    pan  = [(idx[0], idx[1], top3[0], top3[1]),
            (idx[0], idx[2], top3[0], top3[2]),
            (idx[1], idx[2], top3[1], top3[2])]
    best = basis_df.set_index('basis').loc[basis]
    blurb = (f'{basis.upper()} basis — the 3 gated components most informative about subclass '
             f'(' + ', '.join(f'{c} R²={r2s[c]:.2f}' for c in top3) + f'); best plane '
             f'{best["best_plane"]} keeps {best["purity_kept"]:.2f} of this basis\'s '
             f'{best["subspace_purity"]:.3f} subspace purity')
    print(f'  {basis}: panels on {top3}')
    save_archetype_scatter_pdf(
        coords, assign['archetype'].values, pan, aa=None,
        title=f'Jorstad23 human IT joint embedding, by archetype ({n_pooled} cells, '
              f'04 letters)\n{blurb}',
        out_path=OUT_PDF[basis]['archetype'], label_order=ARCH_ORDER, cmap='tab20', s=1)
    save_archetype_scatter_pdf(
        coords, subclasses, pan, aa=None,
        title=f'Jorstad23 human IT joint embedding, by subclass ({n_pooled} cells)\n{blurb}',
        out_path=OUT_PDF[basis]['subclass'], legend_title='subclass',
        label_order=[cfg['human_subclass'] for cfg in SUBCLASSES], cmap='tab10', s=1)

# ---------------------------------------------------------------------------
# 10. The geometry the archetype centroids form
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
    print(f'   {name:12s} ' + '  '.join(f'{c}={P.loc[name, c]:7.2f}' for c in ARC_SPC)
          + f'  angle={angle[name]:7.1f}  r={radius[name]:6.2f} '
            f'({radius[name] / med_radius:.2f}x med)  n={int(n_by_arch.get(name, 0)):6d}{flag}')
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

# arc-free cross-check: the order along the single most cell-type-driven axis
axis1 = ARC_SPC[0]
ax_order = list(P[axis1].sort_values().index)
mean_rank = lambda o, sub: np.mean([i for i, n in enumerate(o) if subclass_of(n) == sub])
if mean_rank(ax_order, DIR_FIRST) > mean_rank(ax_order, DIR_LAST):
    ax_order = ax_order[::-1]
rho, _ = spearmanr([arc.index(n) for n in ARCH_ORDER], [ax_order.index(n) for n in ARCH_ORDER])
print(f'\n   {axis1}-only order: ' + ' -> '.join(ax_order))
print(f'   Spearman rho(arc order, {axis1}-only order) = {rho:.3f}, '
      f'{len(inverted_pairs(arc, ax_order))} inverted pairs')

# is the ordering an artifact of the 50.6/32.7/11.3/5.4 % class imbalance? The written
# centroids use N_TOP_CELLS cells each (equal by construction); these use every assigned
# cell (size-weighted). Agreement means imbalance is not driving the order.
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
      f'{sum(1 for v in cross if v < np.median(within))}/{len(cross)}  '
      f'(a continuum whose boundary archetypes collapse would make this large)')
print('  closest pairs:')
for a, b in sorted(pairs, key=lambda p: D.loc[p])[:12]:
    kind = 'WITHIN' if subclass_of(a) == subclass_of(b) else 'cross '
    print(f'   {kind} {a:12s} {b:12s} d={D.loc[a, b]:6.2f}')

# is the cell cloud connected along the arc, or does it break at subclass boundaries?
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
# 11. The depth-order label map — the deliverable
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

label_meta = {f'{cfg["human_subclass"]} {ALPHABET[k]}': (cfg['token'], cfg['human_subclass'],
                                                         ALPHABET[k])
              for cfg in SUBCLASSES for k in range(cfg['noc'])}
within_rank, arc_rows = {}, []
for rank, label in enumerate(arc_order):
    token, subclass, old_letter = label_meta[label]
    j = within_rank.get(subclass, 0)
    within_rank[subclass] = j + 1
    arc_rows.append({'key': f'{token}_{old_letter}', 'token': token,
                     'human_subclass': subclass, 'old_letter': old_letter,
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
    print(f"   {r['arc_rank']:2d}  {r['old_label']:12s} -> {r['new_label']:13s}"
          f"  (angular rank {r['arc_rank_angular']})")

if not curated:
    print('  curated-vs-angular inversions: NOT APPLICABLE (run 1 — arc_rank IS the angular '
          'rank; there is nothing to compare)')
else:
    inverted = inverted_pairs(arc_order, arc)
    print(f'  curated order inverts {len(inverted)} of '
          f'{len(arc_rows) * (len(arc_rows) - 1) // 2} pairs vs the angular sort: '
          f'{inverted if inverted else "none"}')

print('\nDone.  ARC ORDER STATUS: '
      + ('UNCURATED (run 1) — set ARC_ORDER_CURATED and rerun.' if not curated
         else 'CURATED (run 2).'))
