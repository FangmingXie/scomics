"""The Cux2_-/+ regulon: where it is enriched, how it varies across L2/3, and what Cux2Cre does to it.

Three panels, one PDF (editable text). The first two are yoo25 P21 (normally reared); the third
is the jainlab26 Cux2Cre/WT comparison from script 08.

  A) Dot plot of the Cux2_-/+ enrichment across the three L2/3 archetypes, read from 54's
     expression-stratified test. Colour = log2_enr, dot area = overlap / n_markers (the share of
     the archetype's marker programme the regulon covers), black outline = the 54b star
     criterion, gray fill = overlap < MASK_MIN_OVERLAP (too few shared genes to trust). Encodings
     and thresholds are copied from 54b/54d, except (i) the dot-area reference, rescaled for a
     single-regulon panel (see FRAC_REF), and (ii) the colour ramp, made symmetric about 0 (see
     COLOR_ABS) rather than 54b's off-centre [-1.5, 3.5]. Areas and colours therefore compare
     within this panel, not against 54d; the star and mask rules are unchanged.
  B) Cux2 regulon activity per cell in yoo25 P21 L2/3, binned along the A' -> C' axis. The axis
     is the per-cell score contrast score(A') - score(C') from the script-34 PCHA fit; cells are
     sorted on it and cut into N_BINS EQUAL-SIZED groups (not equal-width, so every box rests on
     the same number of cells), bin 1 the most A'-like and bin N_BINS the most C'-like. Activity
     is the mean over the regulon's target genes of each gene's expression z-scored across these
     cells, so 0 is the L2/3 average and the units are per-gene SDs. Reported with a Spearman
     correlation against the underlying continuous axis -- the bins are for display, the
     correlation does not depend on them.
  C) log2FC (Cux2Cre / WT) of the regulon's target genes in jainlab26 L2/3, as four boxes: all
     targets, then the targets that are also A', B' and C' markers -- i.e. exactly the `overlap`
     sets panel A's three dots are computed from. The fold change is RECOMPUTED here from the
     04_v2 h5ads rather than read from script 08's table, so that all N targets appear: 08 drops
     genes detected in under MIN_FRAC_CELLS of the cells in both genotypes, which costs three
     targets. The estimator is 08's exactly (per-gene mean CP10k, depths from the full matrix,
     PSEUDOCOUNT on the CP10k scale, same cells and same UMAP box), and the script asserts the
     recomputed values reproduce 08's for every gene the two have in common. Genes below the
     detection floor are drawn as OPEN markers -- they are shown, but a fold change built on ~1%
     of cells should not be read like the others. The three archetype boxes are SUBSETS of the
     first, not independent groups, and they are tiny (7 / 3 / 1 genes), so no test is run across
     them. Cux2_-/+ is a REPRESSIVE regulon (TF up, targets down), so if the Cre allele lowers
     Cux2 function these genes should sit ABOVE zero. Box colours are black (all targets) then
     C0/C1/C2 for A'/B'/C', matching panels A and B.

  D) The same boxplot, but the groups are the FULL archetype marker sets -- every A', B' and C'
     marker, whether or not it is a Cux2 target. These three sets are disjoint, so unlike panel C
     a test across them is legitimate (Kruskal-Wallis). Panel C asks what the regulon does; panel
     D asks whether the Cre allele moves the archetype programmes at all.

Archetype letters are the published primed labels throughout: the internal PCHA letters
(archetype_1/2/3 = A/B/C) are relabelled through the it_evo/15 depth-arc table, which for L2/3 is
the reversal A -> C', B -> B', C -> A'. Colour follows the DISPLAYED label (A' -> C0, B' -> C1,
C' -> C2), per scripts/ARCHETYPE_MAPPING.md.

Caveats worth carrying: the regulon is a SCENIC+ call with `extended` (motif-inferred) targets, so
membership is correlational, not validated binding; and panel C's p-value treats genes as
independent, which they are not.

Reads:
  local_data/res/it/54.L2_3_stratified_enrichment.tsv         (panel A)
  local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv         (regulon membership)
  local_data/res/it/34.follow.two_L23_archetype_scores.tsv    (archetype assignment)
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv   (panels C and D gene sets)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv  (primed relabel)
  links/it/superdupermegaRNA_yoo25_IT_P21.h5ad                (panel B expression)
  local_data/res/cux2cre/04_v2.l23_jainlab26_{cux2cre,wt}_labeled.h5ad  (panel C cells)
  local_data/res/cux2cre/00_v2.jainlab26_{cux2cre,wt}_labeled.h5ad      (panel C UMAP filter)
  local_data/res/cux2cre/08.l23_de_wt_vs_cux2cre.tsv          (cross-check only)
Outputs:
  local_data/fig/cux2cre/09.cux2_regulon_targets.pdf
  local_data/res/cux2cre/09.l23_gene_log2fc.tsv               (targets + markers, recomputed)
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import kruskal, spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_ENRICH    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '54.L2_3_stratified_enrichment.tsv')
INPUT_TARGETS   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '40.yoo25_L2_3_regulon_targets.tsv')
INPUT_SCORES    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '34.follow.two_L23_archetype_scores.tsv')
INPUT_MARKERS   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '34.follow.two_L23_archetype_markers.tsv')
INPUT_ARC_ORDER = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo', '15.mouse_IT_joint_archetype_arc_order.tsv')
INPUT_YOO25     = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_P21.h5ad')
INPUT_CUX2CRE_L23 = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_L23      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_wt_labeled.h5ad')
INPUT_CUX2CRE_00  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_00       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
INPUT_DE        = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '08.l23_de_wt_vs_cux2cre.tsv')
OUT_FIG         = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '09.cux2_regulon_targets.pdf')
OUT_TSV         = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '09.l23_gene_log2fc.tsv')

REGULON   = 'Cux2_-/+'
ARC_TOKEN = 'L23'
SUBCLASS  = 'L2/3'
SCORE_COLS = {'A': 'score_A', 'B': 'score_B', 'C': 'score_C'}   # internal PCHA letters
ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']   # archetype_1 -> A, archetype_2 -> B, ...
YOO25_PREFIX = 'yoo25:'    # script-34 scores are prefixed by dataset

# Panel A encodings, held identical to 54b/54d so the figures agree.
MASK_MIN_OVERLAP = 5       # 41b/54b
STAR_FDR         = 0.05    # 54b
STAR_LOG2ENR     = 1.0     # 54b
# 54b's ramp is [-1.5, 3.5], which puts white off-centre. Here the scale is SYMMETRIC about 0,
# so equal enrichment and depletion get equal colour weight — colours are therefore not directly
# comparable to 54b/54c/54d, only the star/mask rules are shared.
COLOR_ABS = 3.5
COLOR_MIN, COLOR_MAX = -COLOR_ABS, COLOR_ABS
# Dot area. 54d uses FRAC_REF=0.40 / SIZE_REF=260 across a whole matrix of regulons; this
# regulon covers at most 3.4% of an archetype's markers, which on that scale is a ~2pt dot.
# The reference is rescaled so a single-regulon panel is legible — so dot AREAS HERE ARE NOT
# COMPARABLE TO 54d, only to each other. Colour and the star/mask rules are unchanged.
FRAC_REF = 0.05
SIZE_REF = 400.0

DE_LOG2FC_COL = 'log2fc_cux2cre_over_wt'
ARCH_COLORS   = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
COLOR_TARGET  = 'black'      # the all-targets box in panel C
N_BINS        = 10           # panel B: equal-sized cell bins along the A' -> C' axis
COLOR_BG      = '#B0B0B0'
MIN_N_FOR_BOX = 5     # below this a boxplot's quartiles are noise; draw a median line instead
# Panel C recomputation — every value must match script 08's, so these mirror 08 exactly.
UMAP_X_MAX     = -4
UMAP_Y_MAX     = 10
PSEUDOCOUNT    = 1.0     # on the CP10k scale, as in 08
MIN_FRAC_CELLS = 0.05    # 08's detection floor; genes under it are drawn open, not dropped
RECOMPUTE_TOL  = 1e-9    # agreement required against 08 on the genes it did test

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _umap_filter(adata_l23, path_00):
    """Keep L2/3 cells inside the 00_v2 UMAP box used by scripts 06 and 08."""
    adata_00 = ad.read_h5ad(path_00)
    umap = pd.DataFrame(adata_00.obsm['X_umap'], index=adata_00.obs_names, columns=['UMAP1', 'UMAP2'])
    u = umap.reindex(adata_l23.obs_names)
    if u.isna().any().any():
        raise ValueError(f'L2/3 barcodes missing from {path_00}')
    mask = (u['UMAP1'] < UMAP_X_MAX) & (u['UMAP2'] < UMAP_Y_MAX)
    return adata_l23[mask.values].copy()


def _gene_cp10k_stats(adata, genes):
    """Per-gene mean CP10k and detection fraction. Depths come from the FULL matrix, as in 08."""
    depths = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
    counts = _to_array(adata[:, genes].X).astype(float)
    cp10k = counts / depths[:, None] * 1e4
    return cp10k.mean(axis=0), (counts > 0).mean(axis=0)


# --- Primed relabel, from the persisted depth-arc table ---
arc = pd.read_csv(INPUT_ARC_ORDER, sep='\t')
arc = arc[arc['token'] == ARC_TOKEN]
if arc.empty:
    raise ValueError(f'token {ARC_TOKEN} absent from {INPUT_ARC_ORDER}')
relabel = dict(zip(arc['old_letter'], arc['new_letter']))       # A -> C', B -> B', C -> A'
primed_order = sorted(relabel.values())                          # A', B', C' = depth order
print(f'Primed relabel for {SUBCLASS}: {relabel}')

# --- Regulon membership ---
targets_tbl = pd.read_csv(INPUT_TARGETS, sep='\t')
targets_tbl = targets_tbl[targets_tbl['regulon'] == REGULON]
if targets_tbl.empty:
    raise ValueError(f'regulon {REGULON} absent from {INPUT_TARGETS}')
targets = sorted(targets_tbl['Gene'].unique())
print(f'{REGULON}: {len(targets)} target genes, sources={sorted(targets_tbl["source"].unique())}')

# --- Panel A data: 54's stratified enrichment for this regulon ---
enr = pd.read_csv(INPUT_ENRICH, sep='\t')
enr = enr[enr['regulon'] == REGULON].set_index('arch_letter')
if not set(primed_order) <= set(enr.index):
    raise ValueError(f'{REGULON} missing archetypes in {INPUT_ENRICH}: '
                     f'{sorted(set(primed_order) - set(enr.index))}')
enr = enr.loc[primed_order]
enr['coverage'] = enr['overlap'] / enr['n_markers']
print('Panel A — 54 stratified enrichment:')
print(enr[['overlap', 'n_markers', 'n_targets', 'exp', 'log2_enr', 'fdr_strat', 'coverage']].to_string())

# --- Panel C gene sets: the targets, and their overlap with each archetype's markers ---
mk = pd.read_csv(INPUT_MARKERS, sep='\t')
marker_sets = {}
for arch in sorted(mk['archetype'].unique()):
    old_letter = ARCHETYPE_LETTERS[int(arch.split('_')[1]) - 1]
    if old_letter not in relabel:
        raise ValueError(f'{arch} ({old_letter}) has no primed label in {INPUT_ARC_ORDER}')
    marker_sets[relabel[old_letter]] = set(mk.loc[mk['archetype'] == arch, 'gene'])
if set(primed_order) - set(marker_sets):
    raise ValueError(f'archetypes missing from {INPUT_MARKERS}: '
                     f'{sorted(set(primed_order) - set(marker_sets))}')

overlap_sets = {}
for lab in primed_order:
    genes = sorted(set(targets) & marker_sets[lab])
    expected = int(enr.loc[lab, 'overlap'])
    if len(genes) != expected:
        raise ValueError(f'{lab} overlap is {len(genes)} genes here but {expected} in '
                         f'{INPUT_ENRICH} — the marker set or regulon has drifted')
    overlap_sets[lab] = genes
    print(f'Panel C — {REGULON} targets that are {lab} markers ({len(genes)}): {genes}')

# --- Panel B data: regulon activity in yoo25 P21 L2/3 ---
print(f'Loading yoo25 P21 ({SUBCLASS} only)...')
yoo = ad.read_h5ad(INPUT_YOO25)
yoo = yoo[yoo.obs['Subclass'] == SUBCLASS].copy()
print(f'  {yoo.n_obs} {SUBCLASS} cells x {yoo.n_vars} genes')

missing = [g for g in targets if g not in yoo.var_names]
if missing:
    raise ValueError(f'{len(missing)} target(s) absent from {INPUT_YOO25}: {missing}')

scores = pd.read_csv(INPUT_SCORES, sep='\t', index_col=0)
scores = scores[scores.index.str.startswith(YOO25_PREFIX)]
scores.index = scores.index.str.removeprefix(YOO25_PREFIX)
scores = scores.reindex(yoo.obs_names)
if scores.isna().any().any():
    raise ValueError(f'{int(scores.isna().any(axis=1).sum())} yoo25 cells missing from {INPUT_SCORES}')

# The A' -> C' axis: the score contrast between the two extreme archetypes. `relabel` maps the
# internal PCHA letter to its primed label, so invert it to find which score column is which.
inv_relabel = {v: k for k, v in relabel.items()}
col_first, col_last = (SCORE_COLS[inv_relabel[primed_order[i]]] for i in (0, -1))
print(f'  {primed_order[0]} -> {primed_order[-1]} axis = {col_first} - {col_last}')
axis_score = (scores[col_first] - scores[col_last]).values

# Equal-sized bins: sort on the axis (most A'-like first) and cut into N_BINS contiguous blocks.
order = np.argsort(-axis_score, kind='stable')
bin_idx = np.empty(len(order), dtype=int)
for b, block in enumerate(np.array_split(order, N_BINS)):
    bin_idx[block] = b

# yoo25 X is already log-normalized (row sums ~3.6e3, non-integer), so z-score per gene only.
expr = _to_array(yoo[:, targets].X).astype(float)
z = (expr - expr.mean(axis=0)) / expr.std(axis=0, ddof=0)
if not np.isfinite(z).all():
    raise ValueError('non-finite z-scores — a target gene has zero variance in these cells')
activity = z.mean(axis=1)

activity_by_bin = [activity[bin_idx == b] for b in range(N_BINS)]
kw_stat, kw_p = kruskal(*activity_by_bin)
rho, rho_p = spearmanr(axis_score, activity)
print(f'Panel B — {REGULON} activity (mean z over {len(targets)} targets) along the '
      f'{primed_order[0]}->{primed_order[-1]} axis, {N_BINS} equal-sized bins:')
for b, vals in enumerate(activity_by_bin):
    print(f'  bin {b + 1:2d}: n={len(vals):5d}  axis=[{axis_score[bin_idx == b].min():+.3f}, '
          f'{axis_score[bin_idx == b].max():+.3f}]  mean={vals.mean():+.3f}  '
          f'median={np.median(vals):+.3f}')
print(f'  Spearman rho = {rho:+.3f} (p = {rho_p:.3g}) against the continuous axis  |  '
      f'Kruskal-Wallis across bins p = {kw_p:.3g}')

# --- Panel C data: recompute the target log2FC from the jainlab26 cells ---
print('Loading jainlab26 L2/3 h5ads...')
cux2cre = _umap_filter(ad.read_h5ad(INPUT_CUX2CRE_L23), INPUT_CUX2CRE_00)
wt      = _umap_filter(ad.read_h5ad(INPUT_WT_L23),      INPUT_WT_00)
print(f'  after the 00_v2 UMAP box — wt: {wt.n_obs} cells  |  cux2cre: {cux2cre.n_obs} cells')
# One recomputation covering every gene either panel needs: the regulon's targets (panel C)
# and the full archetype marker sets (panel D).
#
# The marker sets come from cheng22/yoo25, whose reference predates jainlab26's mm39 symbols, so
# a handful of markers have been renamed and simply cannot be measured here (Arntl, March1,
# Nrd1, ...). Those are DROPPED with a report — no symbol remapping is attempted, because no
# authoritative mapping ships with this project and guessing would silently invent data. A
# missing TARGET is still a hard error: panel C's gene set must match panel A's exactly.
absent_targets = [g for g in targets if g not in wt.var_names]
if absent_targets:
    raise ValueError(f'regulon targets absent from the jainlab26 h5ads: {absent_targets}')

measured_sets = {}
for lab in primed_order:
    kept = sorted(g for g in marker_sets[lab] if g in wt.var_names)
    dropped = sorted(set(marker_sets[lab]) - set(kept))
    if dropped:
        print(f'  NOTE: {lab} — {len(dropped)}/{len(marker_sets[lab])} markers absent from the '
              f'jainlab26 annotation, dropped: {dropped}')
    measured_sets[lab] = kept

gene_union = sorted(set(targets).union(*measured_sets.values()))
print(f'  recomputing log2FC for {len(gene_union)} genes '
      f'({len(targets)} targets + {sum(len(v) for v in measured_sets.values())} measurable '
      f'markers, deduplicated)')

mean_wt,      frac_wt      = _gene_cp10k_stats(wt,      gene_union)
mean_cux2cre, frac_cux2cre = _gene_cp10k_stats(cux2cre, gene_union)
fc = pd.DataFrame({
    'gene':                   gene_union,
    'mean_cp10k_wt':          mean_wt,
    'mean_cp10k_cux2cre':     mean_cux2cre,
    'frac_detected_wt':       frac_wt,
    'frac_detected_cux2cre':  frac_cux2cre,
    DE_LOG2FC_COL: np.log2((mean_cux2cre + PSEUDOCOUNT) / (mean_wt + PSEUDOCOUNT)),
}).set_index('gene')
fc['low_detection'] = (fc['frac_detected_wt'] < MIN_FRAC_CELLS) & (fc['frac_detected_cux2cre'] < MIN_FRAC_CELLS)
fc['is_regulon_target'] = fc.index.isin(targets)
for lab in primed_order:
    fc.loc[fc.index.isin(measured_sets[lab]), 'archetype_marker'] = lab

# The estimator must be 08's, so every gene 08 tested has to come back identical.
de = pd.read_csv(INPUT_DE, sep='\t', index_col=0)
shared = [g for g in gene_union if g in de.index]
delta = np.abs(fc.loc[shared, DE_LOG2FC_COL].values - de.loc[shared, DE_LOG2FC_COL].values).max()
if delta > RECOMPUTE_TOL:
    raise ValueError(f'recomputed log2FC disagrees with {os.path.basename(INPUT_DE)} by {delta:.3g} '
                     f'(tolerance {RECOMPUTE_TOL:g}) — the two estimators have diverged')
print(f'  cross-check vs script 08 on {len(shared)}/{len(gene_union)} genes: max |difference| = {delta:.2g}')
recovered = sorted(set(targets) - set(de.index))
print(f'  recovered {len(recovered)} target(s) that 08 filtered out: {recovered}')
print(f'  recovered {len(set(gene_union) - set(de.index))} gene(s) overall below 08\'s filter')
n_low = int(fc['low_detection'].sum())
print(f'  {n_low} gene(s) below the {MIN_FRAC_CELLS:.0%} detection floor in both genotypes '
      f'(drawn open), of which {int(fc.loc[fc["low_detection"], "is_regulon_target"].sum())} '
      f'are regulon targets')

fc.sort_values(DE_LOG2FC_COL).to_csv(OUT_TSV, sep='\t')
print(f'Saved → {OUT_TSV}')

# Panel C groups: all targets, then each archetype's overlap (subsets of the first)
panel_c = [('all\ntargets', targets, COLOR_TARGET)]
for lab in primed_order:
    panel_c.append((f'{lab}\noverlap', overlap_sets[lab], ARCH_COLORS[lab]))

# Panel D groups: the full archetype marker sets, disjoint, target membership ignored
panel_d = [(f'{lab}\nmarkers', measured_sets[lab], ARCH_COLORS[lab]) for lab in primed_order]
kw_d_stat, kw_d_p = kruskal(*[fc.loc[g, DE_LOG2FC_COL].values for _, g, _ in panel_d])

for name, panel in (('C', panel_c), ('D', panel_d)):
    print(f'Panel {name} — log2FC (Cux2Cre / WT) by group:')
    for lab, genes, _ in panel:
        vals = fc.loc[genes, DE_LOG2FC_COL].values
        print(f'  {lab.replace(chr(10), " "):16s} n={len(vals):4d}  median={np.median(vals):+.4f}  '
              f'mean={vals.mean():+.4f}  up={int((vals > 0).sum())}')
print(f'  panel D Kruskal-Wallis across the three disjoint marker sets: p = {kw_d_p:.3g}')

# --- Figure ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

rng = np.random.default_rng(0)   # jitter for panels B and C
fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2), gridspec_kw={'width_ratios': [1.0, 1.2, 1.15, 0.95]})

# Panel A: single-row dot plot of the stratified enrichment
ax = axes[0]
norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX)
cmap = plt.get_cmap('RdBu_r')
thin = enr['overlap'].values < MASK_MIN_OVERLAP
sig  = ((enr['fdr_strat'].values < STAR_FDR) & (enr['log2_enr'].values > STAR_LOG2ENR)
        & ~thin)
sizes = enr['coverage'].values / FRAC_REF * SIZE_REF
xs = np.arange(len(enr))

ax.scatter(xs[thin], np.zeros(thin.sum()), s=sizes[thin], color=COLOR_BG,
           edgecolor='none', zorder=2)
mappable = ax.scatter(xs[~thin], np.zeros((~thin).sum()), s=sizes[~thin],
                      c=enr['log2_enr'].values[~thin], cmap=cmap, norm=norm,
                      edgecolor='none', zorder=2)
ax.scatter(xs[sig], np.zeros(sig.sum()), s=sizes[sig], facecolor='none',
           edgecolor='black', linewidths=1.2, zorder=3)
for x, (lab, row) in zip(xs, enr.iterrows()):
    ax.annotate(f'{int(row["overlap"])}/{int(row["n_markers"])}', (x, 0),
                textcoords='offset points', xytext=(0, -32), ha='center', fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(enr.index)
ax.set_yticks([0])
ax.set_yticklabels([REGULON])
ax.set_xlim(-0.6, len(enr) - 0.4)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('L2/3 archetype', labelpad=10)
ax.set_title(f'{REGULON} enrichment in L2/3 archetypes\n'
             f'(54 stratified test; yoo25 regulons)', fontsize=9)
ax.text(0.5, 0.97, f'{REGULON} regulon: n = {len(targets)} target genes',
        transform=ax.transAxes, ha='center', va='top', fontsize=7.5, color='dimgray')
ax.tick_params(axis='both', length=0)
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
cb = fig.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.22, fraction=0.05, aspect=24)
cb.set_label('log$_2$ enrichment', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.text(0.5, -0.78, f'dot area = overlap / n markers (rescaled vs 54d)\n'
                    f'outlined: FDR<{STAR_FDR:g} & log$_2$enr>{STAR_LOG2ENR:g};  '
                    f'gray: overlap<{MASK_MIN_OVERLAP}',
        transform=ax.transAxes, ha='center', va='top', fontsize=7, color='dimgray')

# Panel B: regulon activity along the A' -> C' axis, in equal-sized cell bins
ax = axes[1]
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
# Ramp through the three archetype colours so the bins read as A' -> B' -> C'.
bin_cmap = LinearSegmentedColormap.from_list(
    'arc', [to_rgb(ARCH_COLORS[lab]) for lab in primed_order])
bin_colors = [bin_cmap(b / (N_BINS - 1)) for b in range(N_BINS)]
for b, (vals, color) in enumerate(zip(activity_by_bin, bin_colors)):
    ax.scatter(b + rng.uniform(-0.28, 0.28, len(vals)), vals, s=1.5, color=color,
               alpha=0.18, linewidths=0, rasterized=True, zorder=2)
    bp = ax.boxplot(vals, positions=[b], widths=0.62, showfliers=False,
                    patch_artist=True, zorder=3)
    bp['boxes'][0].set(facecolor='white', edgecolor=color, alpha=0.95, linewidth=1.1)
    for key in ('whiskers', 'caps'):
        for line in bp[key]:
            line.set(color=color, linewidth=1.1)
    bp['medians'][0].set(color='black', linewidth=1.4)
ax.set_xticks(range(N_BINS))
ax.set_xticklabels(range(1, N_BINS + 1), fontsize=8)
ax.set_xlim(-0.7, N_BINS - 0.3)
ax.set_xlabel(f"{primed_order[0]}  $\\longleftarrow$  "
              f"{N_BINS} equal-sized bins along score({primed_order[0]}) $-$ "
              f"score({primed_order[-1]})  $\\longrightarrow$  {primed_order[-1]}", fontsize=8)
ax.set_ylabel(f'{REGULON} activity\n(mean z over {len(targets)} target genes)')
ax.set_title(f'Regulon activity along the {primed_order[0]}\u2013{primed_order[-1]} axis, '
             f'yoo25 P21 L2/3\n'
             f'Spearman $\\rho$ = {rho:+.2f} ($n$ = {len(activity)} cells, '
             f'$p$ = {rho_p:.1e})', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

def _log2fc_panel(ax, groups, title, caption=None):
    """Boxplot of log2FC per gene group; groups under MIN_N_FOR_BOX get a bare median line, and
    genes below the detection floor are drawn as open markers."""
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
    for pos, (lab, genes, color) in enumerate(groups):
        vals = fc.loc[genes, DE_LOG2FC_COL].values
        if len(vals) >= MIN_N_FOR_BOX:
            bp = ax.boxplot(vals, positions=[pos], widths=0.55, showfliers=False,
                            patch_artist=True, zorder=2)
            bp['boxes'][0].set(facecolor='white', edgecolor=color, alpha=0.9, linewidth=1.2)
            for key in ('whiskers', 'caps'):
                for line in bp[key]:
                    line.set(color=color, linewidth=1.2)
            bp['medians'][0].set(color='black', linewidth=1.5)
        else:   # too few genes for a box to mean anything — draw the median as a bare line
            ax.hlines(np.median(vals), pos - 0.27, pos + 0.27, color='black',
                      linewidth=1.5, zorder=2)
        low = fc.loc[genes, 'low_detection'].values
        jitter = pos + rng.uniform(-0.16, 0.16, len(vals))
        point_size = 16 if len(vals) < 50 else 4
        ax.scatter(jitter[~low], vals[~low], s=point_size, color=color,
                   alpha=0.85 if len(vals) < 50 else 0.35, edgecolor='white',
                   linewidths=0.4 if len(vals) < 50 else 0, zorder=3, rasterized=len(vals) >= 50)
        ax.scatter(jitter[low], vals[low], s=point_size + 6, facecolor='none', edgecolor=color,
                   linewidths=1.0, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{lab}\nn = {len(genes)}' for lab, genes, _ in groups], fontsize=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel('log$_2$FC  (Cux2Cre / WT)')
    ax.set_title(title, fontsize=9)
    if caption:
        ax.text(0.5, -0.27, caption, transform=ax.transAxes, ha='center', va='top',
                fontsize=7, color='dimgray')
    ax.spines[['top', 'right']].set_visible(False)


OPEN_NOTE = f'open marker: detected in <{MIN_FRAC_CELLS:.0%} of cells in both genotypes'

# Panel C: the regulon's targets and their archetype overlaps (subsets of the first box)
_log2fc_panel(axes[2], panel_c,
              f'{REGULON} target response to Cux2Cre in L2/3\n'
              f"(jainlab26, recomputed; A'/B'/C' boxes are subsets of the first)",
              caption=OPEN_NOTE)

# Panel D: the full archetype marker sets, regardless of regulon membership
_log2fc_panel(axes[3], panel_d,
              f'All L2/3 archetype markers under Cux2Cre\n'
              f'(disjoint sets; Kruskal-Wallis $p$ = {kw_d_p:.2g})',
              caption=OPEN_NOTE)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
print(f'Saved → {OUT_FIG}')
print('Done.')
