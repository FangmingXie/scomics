"""Cux2 expression and the genome-wide differential test in L2/3, jainlab26_wt vs jainlab26_cux2cre.

Three panels, one PDF (editable text):
  A) Per-cell Cux2 expression (log2(1+CP10k)) in L2/3 cells, WT vs Cux2Cre, as boxplots with
     jittered cells behind them. Annotated with n, detection rate, and a two-sided
     Mann-Whitney U p-value.
  B) Cux2 pseudobulk log2 fold change (Cux2Cre / WT) with a bootstrap 95% CI over cells.
  C) Volcano over all genes detected in at least MIN_FRAC_CELLS of the cells in either group:
     pseudobulk log2FC against the BH-adjusted Mann-Whitney p-value. Cux2 is marked in red and
     the top TOP_N_LABEL genes by adjusted p-value are labeled.
  D) The same volcano, recolored by L2/3 archetype marker membership. The marker sets are the
     script-34 PCHA markers (archetype_1/2/3 = internal letters A/B/C), relabeled to the
     published primed letters through the it_evo/15 depth-arc table and colored by the
     DISPLAYED label (A' -> C0, B' -> C1, C' -> C2), per scripts/ARCHETYPE_MAPPING.md. For L2/3
     the relabel is a reversal: A -> C', B -> B', C -> A'.

The WT and Cux2Cre mice are of different sex, so EXCLUDE_GENES (chrY, Xist, Tsix) is
dropped before testing — those are the strongest hits in the comparison and have nothing to do with the
Cre allele. The list is curated by gene symbol: neither the 10x h5 nor the h5ads carry a
chromosome annotation, so it cannot be derived from the data. Genes that survive the filter but
still look sex-driven (near-total loss in one genotype) are printed as a warning, not dropped.

All p-values are descriptive: the test treats cells as independent replicates, which they are
not (one mouse per genotype), so it measures cell-level separation, not between-animal
significance.

Cells are the L2/3 neurons from 04_v2, restricted by the same 00_v2 UMAP spatial filter that
06 uses, so the two groups are the same cell set compared there.

Reads:
  local_data/res/cux2cre/04_v2.l23_jainlab26_{cux2cre,wt}_labeled.h5ad
  local_data/res/cux2cre/00_v2.jainlab26_{cux2cre,wt}_labeled.h5ad   (UMAP filter)
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv          (archetype marker sets)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv    (primed relabel)
Outputs:
  local_data/fig/cux2cre/08.l23_cux2_expr_log2fc.pdf
  local_data/res/cux2cre/08.l23_de_wt_vs_cux2cre.tsv
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_CUX2CRE_L23 = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_L23      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '04_v2.l23_jainlab26_wt_labeled.h5ad')
INPUT_CUX2CRE_00  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_cux2cre_labeled.h5ad')
INPUT_WT_00       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '00_v2.jainlab26_wt_labeled.h5ad')
INPUT_MARKERS     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '34.follow.two_L23_archetype_markers.tsv')
INPUT_ARC_ORDER   = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo', '15.mouse_IT_joint_archetype_arc_order.tsv')
OUT_FIG           = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'cux2cre', '08.l23_cux2_expr_log2fc.pdf')
OUT_TSV           = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'cux2cre', '08.l23_de_wt_vs_cux2cre.tsv')

GENE           = 'Cux2'
UMAP_X_MAX     = -4      # same spatial filter as 06
UMAP_Y_MAX     = 10
PSEUDOCOUNT    = 1.0     # on the CP10k scale
N_BOOT         = 1000
MIN_FRAC_CELLS = 0.05    # gene kept if detected in >= this fraction of cells in either group
# Sex-linked genes dropped before testing (WT and Cux2Cre mice differ in sex). Curated by
# symbol — no chromosome annotation ships with the counts. chrY set, plus Xist and its antisense
# Tsix. Gm52785 is NOT here: the 10x feature order follows the reference GTF, and it sits
# immediately next to Cux2 (chr5 block Ccdc63/Myl2/.../Gm52785/Cux2/Pheta1), not in the chrY
# block at indices 423-453. It is a locus neighbour, not a sex gene, so it is tested and
# highlighted alongside Cux2.
EXCLUDE_GENES  = ['Ddx3y', 'Eif2s3y', 'Kdm5d', 'Uty', 'Usp9y', 'Zfy1', 'Zfy2', 'Sry',
                  'Ube1y1', 'Rbmy', 'Tspy1', 'Xist', 'Tsix']
SEX_LIKE_RATIO = 0.15    # warn if a surviving gene keeps < this fraction of its mean in one group
FDR_CUTOFF     = 0.05
LOG2FC_CUTOFF  = 0.25
TOP_N_LABEL    = 12
COLOR_WT       = '#4C72B0'
COLOR_CUX2CRE  = '#C44E52'
COLOR_SIG      = '#55A868'
LOCUS_GENE     = 'Gm52785'   # transcript adjacent to Cux2 in the reference; marked in both volcanoes
COLOR_LOCUS    = '#8172B2'
ARC_TOKEN      = 'L23'   # key into the it_evo/15 depth-arc table
# archetype_1 -> A, archetype_2 -> B, ... ; the primed relabel comes from INPUT_ARC_ORDER.
ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
# Color follows the DISPLAYED (primed) label, never the internal key — scripts/ARCHETYPE_MAPPING.md.
ARCH_COLORS    = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)


def _to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


def _umap_filter(adata_l23, path_00):
    """Keep L2/3 cells inside the 00_v2 UMAP box used by script 06."""
    adata_00 = ad.read_h5ad(path_00)
    umap = pd.DataFrame(adata_00.obsm['X_umap'], index=adata_00.obs_names, columns=['UMAP1', 'UMAP2'])
    u = umap.reindex(adata_l23.obs_names)
    if u.isna().any().any():
        raise ValueError(f'L2/3 barcodes missing from {path_00}')
    mask = (u['UMAP1'] < UMAP_X_MAX) & (u['UMAP2'] < UMAP_Y_MAX)
    return adata_l23[mask.values].copy()


def _bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def _stagger_labels(x, y, labels, y_span, x_span, x_pad=0.025, min_sep=0.04):
    """Place labels to the right of their points, pushed apart vertically so they do not overlap.

    Returns (label, label_x, label_y, point_x, point_y) tuples. Labels are processed bottom-up;
    each is nudged above the previous one whenever they would sit closer than `min_sep` of the
    y-range apart.
    """
    order = np.argsort(y)
    sep = min_sep * y_span
    placed = []
    last_y = -np.inf
    for i in order:
        ly = max(y[i], last_y + sep)
        last_y = ly
        placed.append((labels[i], x[i] + x_pad * x_span, ly, x[i], y[i]))
    return placed


# --- Load L2/3 cells ---
print('Loading L2/3 h5ads...')
cux2cre = ad.read_h5ad(INPUT_CUX2CRE_L23)
wt      = ad.read_h5ad(INPUT_WT_L23)
print(f'  cux2cre: {cux2cre.n_obs} cells  |  wt: {wt.n_obs} cells')

print(f'Applying 00_v2 UMAP filter (x < {UMAP_X_MAX}, y < {UMAP_Y_MAX})...')
cux2cre = _umap_filter(cux2cre, INPUT_CUX2CRE_00)
wt      = _umap_filter(wt,      INPUT_WT_00)
print(f'  After filter — cux2cre: {cux2cre.n_obs}  |  wt: {wt.n_obs}')

if not (cux2cre.var_names == wt.var_names).all():
    raise ValueError('cux2cre and wt var_names differ — cannot compare gene-wise')
if GENE not in wt.var_names:
    raise ValueError(f'Gene {GENE} not found in var_names')

# --- Gene filter, then CP10k normalization on the kept genes ---
counts_wt      = _to_array(wt.X)
counts_cux2cre = _to_array(cux2cre.X)
depths_wt      = counts_wt.sum(axis=1, keepdims=True)
depths_cux2cre = counts_cux2cre.sum(axis=1, keepdims=True)

frac_wt      = (counts_wt      > 0).mean(axis=0)
frac_cux2cre = (counts_cux2cre > 0).mean(axis=0)
keep = (frac_wt >= MIN_FRAC_CELLS) | (frac_cux2cre >= MIN_FRAC_CELLS)
print(f'Genes detected in >= {MIN_FRAC_CELLS:.0%} of cells in either group: {keep.sum()} / {len(keep)}')

excluded_present = [g for g in EXCLUDE_GENES if g in wt.var_names[keep]]
keep &= ~np.isin(wt.var_names, EXCLUDE_GENES)
print(f'Sex-linked genes excluded: {len(excluded_present)} of {len(EXCLUDE_GENES)} present — {excluded_present}')
genes_kept = wt.var_names[keep]
print(f'Genes tested: {keep.sum()}')
if GENE not in genes_kept:
    raise ValueError(f'Gene {GENE} did not pass the detection filter')

cp10k_wt      = counts_wt[:,      keep] / depths_wt      * 1e4
cp10k_cux2cre = counts_cux2cre[:, keep] / depths_cux2cre * 1e4
log_wt        = np.log2(1 + cp10k_wt)
log_cux2cre   = np.log2(1 + cp10k_cux2cre)

# --- Panel A values: per-cell GENE expression ---
gi = genes_kept.get_loc(GENE)
gene_log_wt      = log_wt[:, gi]
gene_log_cux2cre = log_cux2cre[:, gi]
det_wt      = (cp10k_wt[:, gi]      > 0).mean()
det_cux2cre = (cp10k_cux2cre[:, gi] > 0).mean()

# --- Genome-wide differential test (vectorized over genes) ---
print('Running Mann-Whitney U over all kept genes...')
_, p_vals = mannwhitneyu(log_wt, log_cux2cre, alternative='two-sided', axis=0)
p_adj = _bh_fdr(p_vals)

mean_wt      = cp10k_wt.mean(axis=0)
mean_cux2cre = cp10k_cux2cre.mean(axis=0)
log2fc = np.log2((mean_cux2cre + PSEUDOCOUNT) / (mean_wt + PSEUDOCOUNT))

de = pd.DataFrame({
    'gene':                   genes_kept,
    'mean_cp10k_wt':          mean_wt,
    'mean_cp10k_cux2cre':     mean_cux2cre,
    'log2fc_cux2cre_over_wt': log2fc,
    'pval':                   p_vals,
    'padj':                   p_adj,
}).set_index('gene')
de.sort_values('padj').to_csv(OUT_TSV, sep='\t')
print(f'Saved → {OUT_TSV}')

n_sig = ((de['padj'] < FDR_CUTOFF) & (de['log2fc_cux2cre_over_wt'].abs() > LOG2FC_CUTOFF)).sum()
print(f'Significant (FDR < {FDR_CUTOFF}, |log2FC| > {LOG2FC_CUTOFF}): {n_sig}')

# Surviving genes that behave like sex genes: near-total loss in one genotype. Reported only —
# without a chromosome annotation they cannot be assigned to chrY automatically.
ratio = np.minimum(de['mean_cp10k_wt'], de['mean_cp10k_cux2cre']) / np.maximum(
    np.maximum(de['mean_cp10k_wt'], de['mean_cp10k_cux2cre']), np.finfo(float).tiny)
sex_like = de[(ratio < SEX_LIKE_RATIO) & (de['padj'] < FDR_CUTOFF)].sort_values('padj')
if len(sex_like):
    print(f'WARNING: {len(sex_like)} significant gene(s) keep < {SEX_LIKE_RATIO:.0%} of their '
          f'expression in one genotype — check whether they are sex-linked:')
    print(sex_like[['mean_cp10k_wt', 'mean_cp10k_cux2cre', 'padj']].head(10).to_string())

gene_p    = de.loc[GENE, 'pval']
gene_padj = de.loc[GENE, 'padj']
gene_fc   = de.loc[GENE, 'log2fc_cux2cre_over_wt']
print(f'{GENE}: wt mean={gene_log_wt.mean():.3f} (det {det_wt:.1%}, n={len(gene_log_wt)})  |  '
      f'cux2cre mean={gene_log_cux2cre.mean():.3f} (det {det_cux2cre:.1%}, n={len(gene_log_cux2cre)})')
print(f'{GENE}: log2FC={gene_fc:.3f}  p={gene_p:.3g}  padj={gene_padj:.3g}  |  '
      f'mean CP10k — wt: {mean_wt[gi]:.2f}, cux2cre: {mean_cux2cre[gi]:.2f}')

# --- Bootstrap CI on the GENE fold change (resampling cells within each group) ---
def _log2fc_gene(a_wt, a_cux2cre):
    return np.log2((a_cux2cre.mean() + PSEUDOCOUNT) / (a_wt.mean() + PSEUDOCOUNT))

rng = np.random.default_rng(0)
boot = np.array([_log2fc_gene(rng.choice(cp10k_wt[:, gi],      len(cp10k_wt),      replace=True),
                              rng.choice(cp10k_cux2cre[:, gi], len(cp10k_cux2cre), replace=True))
                 for _ in range(N_BOOT)])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f'{GENE} log2FC bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')

# --- Figure ---
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={'width_ratios': [1.2, 0.75, 1.6, 1.6]})

# Panel A: boxplot of per-cell GENE expression
ax = axes[0]
groups = [('WT', gene_log_wt, COLOR_WT, det_wt), ('Cux2Cre', gene_log_cux2cre, COLOR_CUX2CRE, det_cux2cre)]
rng_jitter = np.random.default_rng(0)
for pos, (label, vals, color, det) in enumerate(groups):
    ax.scatter(pos + rng_jitter.uniform(-0.18, 0.18, len(vals)), vals,
               s=2, color=color, alpha=0.15, linewidths=0, rasterized=True, zorder=1)
    bp = ax.boxplot(vals, positions=[pos], widths=0.5, showfliers=False,
                    patch_artist=True, zorder=2)
    bp['boxes'][0].set(facecolor='white', edgecolor=color, alpha=0.9, linewidth=1.2)
    for key in ('whiskers', 'caps'):
        for line in bp[key]:
            line.set(color=color, linewidth=1.2)
    bp['medians'][0].set(color='black', linewidth=1.5)

y_top = max(gene_log_wt.max(), gene_log_cux2cre.max())
ax.plot([0, 0, 1, 1], [y_top * 1.04, y_top * 1.09, y_top * 1.09, y_top * 1.04], color='black', linewidth=1.0)
ax.text(0.5, y_top * 1.10, f'Mann-Whitney $p$ = {gene_p:.1e}', ha='center', va='bottom', fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'WT\nn = {len(gene_log_wt)}\n{det_wt:.0%} detected',
                    f'Cux2Cre\nn = {len(gene_log_cux2cre)}\n{det_cux2cre:.0%} detected'])
ax.set_ylabel(f'{GENE} expression  log$_2$(1 + CP10k)')
ax.set_title(f'{GENE} in L2/3 cells', fontsize=10)
ax.set_ylim(-0.3, y_top * 1.22)
ax.spines[['top', 'right']].set_visible(False)

# Panel B: GENE log2 fold change
ax = axes[1]
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
ax.bar([0], [gene_fc], width=0.5, color=COLOR_CUX2CRE, alpha=0.85, zorder=2)
ax.errorbar([0], [gene_fc], yerr=[[gene_fc - ci_lo], [ci_hi - gene_fc]],
            fmt='none', ecolor='black', capsize=4, linewidth=1.2, zorder=3)
ax.text(0, ci_lo - 0.012, f'{gene_fc:.2f}', ha='center', va='top', fontsize=9)
ax.set_xticks([0])
ax.set_xticklabels([f'{GENE}'])
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(min(ci_lo * 1.8, -0.05), max(ci_hi * 1.8, 0.05))
ax.set_ylabel('log$_2$FC  (Cux2Cre / WT)')
ax.set_title(f'{GENE} fold change in L2/3', fontsize=10)
ax.text(0.5, 0.015, f'error bar: bootstrap 95% CI\n({N_BOOT} resamples of cells)',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color='dimgray')
ax.spines[['top', 'right']].set_visible(False)

# --- L2/3 archetype marker sets, relabeled to the published primed letters ---
mk = pd.read_csv(INPUT_MARKERS, sep='\t')
arc = pd.read_csv(INPUT_ARC_ORDER, sep='\t')
arc = arc[arc['token'] == ARC_TOKEN]
if arc.empty:
    raise ValueError(f'token {ARC_TOKEN} absent from {INPUT_ARC_ORDER}')
relabel = dict(zip(arc['old_letter'], arc['new_letter']))   # A -> C', B -> B', C -> A'

marker_sets = {}
for arch in sorted(mk['archetype'].unique()):
    old_letter = ARCHETYPE_LETTERS[int(arch.split('_')[1]) - 1]
    if old_letter not in relabel:
        raise ValueError(f'{arch} ({old_letter}) has no primed label in {INPUT_ARC_ORDER}')
    marker_sets[relabel[old_letter]] = set(mk.loc[mk['archetype'] == arch, 'gene'])
marker_sets = dict(sorted(marker_sets.items()))
overlap = set.intersection(*marker_sets.values()) if len(marker_sets) > 1 else set()
if overlap:
    raise ValueError(f'archetype marker sets are not disjoint: {sorted(overlap)}')
print('Archetype marker sets (primed label -> n markers, n tested here):')
for lab, genes in marker_sets.items():
    in_de = de.index.isin(genes)
    n_sig_arch = ((de['padj'].values < FDR_CUTOFF) &
                  (np.abs(de['log2fc_cux2cre_over_wt'].values) > LOG2FC_CUTOFF) & in_de).sum()
    print(f'  {lab}: {len(genes)} markers, {in_de.sum()} tested, {n_sig_arch} significant, '
          f'median log2FC = {np.median(de.loc[in_de, "log2fc_cux2cre_over_wt"]):+.3f}')
print(f'  {GENE} is a marker of: '
      f'{[lab for lab, g in marker_sets.items() if GENE in g] or "none"}')

# Panels C and D: the same volcano, colored two ways
neglog_padj = -np.log10(np.maximum(de['padj'].values, np.finfo(float).tiny))
fc_vals     = de['log2fc_cux2cre_over_wt'].values
sig = (de['padj'].values < FDR_CUTOFF) & (np.abs(fc_vals) > LOG2FC_CUTOFF)


def _volcano_frame(ax):
    """Cutoff guides, axis labels and the Cux2 / locus-neighbour markers — shared by both volcanoes."""
    ax.axhline(-np.log10(FDR_CUTOFF), color='gray', linewidth=0.7, linestyle='--', zorder=1)
    for cutoff in (-LOG2FC_CUTOFF, LOG2FC_CUTOFF):
        ax.axvline(cutoff, color='gray', linewidth=0.7, linestyle='--', zorder=1)
    for name, color, dy in ((GENE, COLOR_CUX2CRE, -4), (LOCUS_GENE, COLOR_LOCUS, -4)):
        x = de.loc[name, 'log2fc_cux2cre_over_wt']
        y = -np.log10(max(de.loc[name, 'padj'], np.finfo(float).tiny))
        ax.scatter([x], [y], s=45, facecolor='none', edgecolor=color, linewidths=1.5, zorder=6)
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(8, dy),
                    fontsize=9, color=color, fontweight='bold', zorder=7)
    ax.set_xlabel('log$_2$FC  (Cux2Cre / WT)')
    ax.set_ylabel('$-$log$_{10}$ FDR')
    ax.spines[['top', 'right']].set_visible(False)


# Panel C: colored by significance
ax = axes[2]
_volcano_frame(ax)
ax.scatter(fc_vals[~sig], neglog_padj[~sig],
           s=4, color='lightgray', linewidths=0, rasterized=True, zorder=2)
ax.scatter(fc_vals[sig], neglog_padj[sig],
           s=5, color=COLOR_SIG, alpha=0.7, linewidths=0, rasterized=True, zorder=3)

top = de.loc[~de.index.isin([GENE, LOCUS_GENE])].sort_values('padj').head(TOP_N_LABEL)
top_y = -np.log10(np.maximum(top['padj'].values, np.finfo(float).tiny))
for label, lx, ly, px, py in _stagger_labels(top['log2fc_cux2cre_over_wt'].values, top_y, top.index.values,
                                             y_span=np.ptp(neglog_padj), x_span=np.ptp(fc_vals)):
    ax.plot([px, lx], [py, ly], color='gray', linewidth=0.4, zorder=4)
    ax.text(lx, ly, label, fontsize=6.5, color='black', va='center', ha='left', zorder=5)
ax.set_title(f'All genes — {n_sig} significant at FDR < {FDR_CUTOFF}, '
             f'|log$_2$FC| > {LOG2FC_CUTOFF}\n'
             f'({keep.sum()} tested; chrY, Xist, Tsix excluded)', fontsize=9)

# Panel D: the same volcano, colored by L2/3 archetype marker membership
ax = axes[3]
_volcano_frame(ax)
is_marker = np.zeros(len(de), dtype=bool)
for genes in marker_sets.values():
    is_marker |= de.index.isin(genes)
ax.scatter(fc_vals[~is_marker], neglog_padj[~is_marker],
           s=4, color='lightgray', linewidths=0, rasterized=True, zorder=2)
for lab, genes in marker_sets.items():
    m = de.index.isin(genes)
    ax.scatter(fc_vals[m], neglog_padj[m], s=9, color=ARCH_COLORS[lab], alpha=0.85,
               linewidths=0, zorder=3, label=f'{lab}  (n = {m.sum()})')
ax.legend(title='L2/3 archetype markers', fontsize=7, title_fontsize=7,
          loc='upper left', bbox_to_anchor=(0.0, 0.86),   # clear of the Gm52785 label
          frameon=False, handletextpad=0.3)
ax.set_title('Same volcano, colored by L2/3 archetype marker set\n'
             '(scripts/it 34 markers; primed labels from it_evo/15)', fontsize=9)
ax.set_xlim(axes[2].get_xlim())   # same frame as panel C, which the gene labels widened
ax.set_ylim(axes[2].get_ylim())

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches='tight', dpi=300)
print(f'Saved → {OUT_FIG}')
print('Done.')
