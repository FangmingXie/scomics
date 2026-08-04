"""Per-bin boxplots of regulon-target mean expression along PC1 — mouse Cheng22 & human Jorstad23 L2/3 IT.

For each selected regulon, the per-cell score is the MEAN across the regulon's activated (+/+)
target genes of each gene's min-max-normalized expression (each gene scaled to [0, 1] using its
0th–99th expression percentile BEFORE averaging, so highly-expressed genes don't dominate the
mean). That score is shown as boxplots across 10 PC1 bins, mouse and human side by side (one
regulon per page). Mouse and human panels use INDEPENDENT y-axes (not shared).

Regulons are SPECIES-SPECIFIC (each species uses its own SCENIC+ regulon): mouse from yoo25,
human from Wang25. Selected TFs: Meis2, Nfia, Nfib, Rfx3, Satb1, Jdp2 (both species) and Maf
(human-only — no mouse regulon exists, so its mouse panel is omitted and the layout adapts).
Only +/+ (activated) targets are used. Target sets are intersected with each species' gene
universe; coverage is printed (no fail-fast on regulon targets).

Expression (both species, log2(CP10k+1) as in scripts 21/40, then per-gene min-max [0,1]):
  - Mouse h5ad: raw counts in X       (subset Subclass == 'L2/3').
  - Human h5ad: raw counts in .raw.X.

Reads:
  local_data/res/l23_evo/21.mouse_pcha_xp.tsv
  local_data/res/l23_evo/25.human_pcha_xp.tsv
  local_data/res/it/40.yoo25_L2_3_regulon_targets.tsv          (mouse regulons)
  local_data/res/l23_evo/27.human_wang25_regulon_targets.tsv    (human regulons)
  links/l23_evo/cheng22_mouse_IT_P28.h5ad
  links/l23_evo/jorstad23_human_WithinArea_L23IT.h5ad
Outputs:
  local_data/fig/l23_evo/42.pc1_regulon_expr_boxbins_mouse_human.pdf
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_MOUSE_XP  = os.path.join(OUT_RES_DIR, '21.mouse_pcha_xp.tsv')
IN_HUMAN_XP  = os.path.join(OUT_RES_DIR, '25.human_pcha_xp.tsv')
IN_MOUSE_REG = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it', '40.yoo25_L2_3_regulon_targets.tsv')
IN_HUMAN_REG = os.path.join(OUT_RES_DIR, '27.human_wang25_regulon_targets.tsv')
INPUT_MOUSE  = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'cheng22_mouse_IT_P28.h5ad')
INPUT_HUMAN  = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'jorstad23_human_WithinArea_L23IT.h5ad')
OUT_PDF_BOX  = os.path.join(OUT_FIG_DIR, '42.pc1_regulon_expr_boxbins_mouse_human.pdf')

# --- parameters ---
MOUSE_SUBCLASS = 'L2/3'
# Regulon selection: per-species regulon key in each species' regulon table (None = absent).
REGULONS = [
    {'name': 'Meis2', 'mouse': 'Meis2_+/+', 'human': 'MEIS2_+/+'},
    {'name': 'Nfia',  'mouse': 'Nfia_+/+',  'human': 'NFIA_+/+'},
    {'name': 'Nfib',  'mouse': 'Nfib_+/+',  'human': 'NFIB_+/+'},
    {'name': 'Rfx3',  'mouse': 'Rfx3_+/+',  'human': 'RFX3_+/+'},
    {'name': 'Satb1', 'mouse': 'Satb1_+/+', 'human': 'SATB1_+/+'},
    {'name': 'Jdp2',  'mouse': 'Jdp2_+/+',  'human': 'JDP2_+/+'},
    {'name': 'Maf',   'mouse': None,        'human': 'MAF_+/+'},   # no mouse regulon
]
# Display-only PC1/PC2 sign flips, matching archetype figures 21.viz / 25.viz.
MOUSE_PC1_SIGN = 1.0    # mouse FLIP = [1, -1] -> PC1 unchanged, PC2 flipped
MOUSE_PC2_SIGN = -1.0
HUMAN_PC1_SIGN = -1.0   # human FLIP = [-1, 1] -> PC1 flipped, PC2 unchanged
HUMAN_PC2_SIGN = 1.0
DPI            = 300
N_PC1_BINS     = 10
NORM_PCTILE    = (0, 99)   # per-gene min-max normalization percentiles (before averaging)
MOUSE_COLOR    = '#2166ac'
HUMAN_COLOR    = '#b2182b'
FILL_ALPHA     = 0.3
YLABEL         = 'mean min-max target expr [0–1]'
MOUSE_TITLE    = 'Cheng22 mouse L2/3 IT'
HUMAN_TITLE    = 'Jorstad23 human L2/3 IT'

os.makedirs(OUT_FIG_DIR, exist_ok=True)


def binned_groups(x, y, n_bins):
    """Split y into n_bins lists by equal-width bins over x's own [min, max] range.

    Returns (positions, groups) for non-empty bins only: positions are 1-based bin indices
    (1..n_bins) and groups[i] is the array of y-values in that bin (for ax.boxplot).
    """
    edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    idx   = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    positions, groups = [], []
    for b in range(n_bins):
        vals = y[idx == b]
        if vals.size:
            positions.append(b + 1)
            groups.append(vals)
    return positions, groups


def draw_boxbins(ax, positions, groups, color, title):
    """Draw per-bin boxplots (fliers hidden) colored by species onto ax."""
    bp = ax.boxplot(groups, positions=positions, widths=0.6, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(FILL_ALPHA)
        patch.set_edgecolor(color)
    for part in ('whiskers', 'caps'):
        for line in bp[part]:
            line.set_color(color)
    for median in bp['medians']:
        median.set_color('black')
    ax.set_xticks(range(1, N_PC1_BINS + 1))
    ax.set_xticklabels(range(1, N_PC1_BINS + 1))
    ax.set_xlabel(f'PC1 bin (1..{N_PC1_BINS}, low→high)')
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    sns.despine(ax=ax)


def load_regulon_scores(h5ad_path, regulon_targets, cell_index, use_raw, subclass=None):
    """Per-cell mean of each regulon's target genes (per-gene min-max normalized), aligned to cell_index.

    Each target gene's log2(CP10k+1) expression is min-max scaled to [0, 1] using its own
    NORM_PCTILE (0th–99th) percentiles BEFORE averaging across the regulon's genes, so
    highly-expressed genes do not dominate the mean.

    regulon_targets: dict regname -> set of target symbols in this species' casing.
    use_raw: read raw counts from adata.raw.X (human) vs adata.X (mouse).
    Returns (scores: dict regname -> per-cell mean Series, coverage: dict regname -> (n_present, n_total)).
    Target genes absent from the species' panel are dropped (coverage reports how many remain).
    """
    adata = ad.read_h5ad(h5ad_path)
    if subclass is not None:
        adata = adata[adata.obs['Subclass'] == subclass].copy()

    if use_raw:
        if adata.raw is None:
            raise ValueError('h5ad has no .raw; cannot recompute log2(CP10k+1) from raw counts.')
        var_names = (adata.raw.var['feature_name'].values
                     if 'feature_name' in adata.raw.var.columns else adata.raw.var_names.values)
        X_raw = adata.raw.X.toarray().astype(np.float32)
    else:
        var_names = adata.var_names.values
        X_raw = adata.X.toarray().astype(np.float32)

    name2idx = {g: i for i, g in enumerate(var_names)}
    depths = X_raw.sum(axis=1, keepdims=True)
    depths[depths == 0] = 1

    def minmax01(v):
        lo, hi = np.percentile(v, NORM_PCTILE)
        rng = hi - lo if hi > lo else 1.0
        return np.clip((v - lo) / rng, 0.0, 1.0)

    present = {reg: [g for g in tgs if g in name2idx] for reg, tgs in regulon_targets.items()}
    union = sorted({g for gs in present.values() for g in gs})
    expr = pd.DataFrame(
        {g: minmax01(np.log2(X_raw[:, name2idx[g]] / depths[:, 0] * 1e4 + 1)) for g in union},
        index=adata.obs_names.values,
    )

    scores, coverage = {}, {}
    for reg, tgs in regulon_targets.items():
        genes = present[reg]
        coverage[reg] = (len(genes), len(tgs))
        s = expr[genes].mean(axis=1).reindex(cell_index)
        if s.isna().any():
            raise ValueError(f'{reg}: NaN after reindexing to PC1 barcodes (barcode mismatch).')
        scores[reg] = s
    return scores, coverage


# --- load cached PC1/PC2 (display-flipped to match archetype figures) ---
mouse_xp = pd.read_csv(IN_MOUSE_XP, sep='\t', index_col=0)
human_xp = pd.read_csv(IN_HUMAN_XP, sep='\t', index_col=0)
mouse_pc1 = mouse_xp['PC1'].values * MOUSE_PC1_SIGN
mouse_pc2 = mouse_xp['PC2'].values * MOUSE_PC2_SIGN
human_pc1 = human_xp['PC1'].values * HUMAN_PC1_SIGN
human_pc2 = human_xp['PC2'].values * HUMAN_PC2_SIGN

# --- regulon target sets (+/+ only), per species ---
mreg = pd.read_csv(IN_MOUSE_REG, sep='\t')
hreg = pd.read_csv(IN_HUMAN_REG, sep='\t')


def targets(df, key):
    return set(df.loc[df['regulon'] == key, 'Gene'])


mouse_targets = {r['name']: targets(mreg, r['mouse']) for r in REGULONS if r['mouse'] is not None}
human_targets = {r['name']: targets(hreg, r['human']) for r in REGULONS if r['human'] is not None}

# --- per-cell regulon mean-target scores, aligned to PC1 barcodes ---
print('Loading mouse regulon target expression...')
mouse_scores, mouse_cov = load_regulon_scores(INPUT_MOUSE, mouse_targets, mouse_xp.index,
                                              use_raw=False, subclass=MOUSE_SUBCLASS)
print('Loading human regulon target expression...')
human_scores, human_cov = load_regulon_scores(INPUT_HUMAN, human_targets, human_xp.index, use_raw=True)

for r in REGULONS:
    name = r['name']
    if name in mouse_cov:
        print(f'  {name} mouse: {mouse_cov[name][0]}/{mouse_cov[name][1]} targets used')
    else:
        print(f'  {name} mouse: (no mouse regulon — skipped)')
    if name in human_cov:
        print(f'  {name} human: {human_cov[name][0]}/{human_cov[name][1]} targets used')


def panels_for(name):
    """Ordered list of present-species panel descriptors for a regulon."""
    out = []
    if name in mouse_scores:
        out.append(dict(sp='mouse', pc1=mouse_pc1, pc2=mouse_pc2, score=mouse_scores[name].values,
                        color=MOUSE_COLOR, title=MOUSE_TITLE, n=mouse_cov[name][0]))
    if name in human_scores:
        out.append(dict(sp='human', pc1=human_pc1, pc2=human_pc2, score=human_scores[name].values,
                        color=HUMAN_COLOR, title=HUMAN_TITLE, n=human_cov[name][0]))
    return out


plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- per-bin boxplots (mouse left, human right; independent y-axes) ---
print(f'Writing {OUT_PDF_BOX} ({len(REGULONS)} pages)...')
with PdfPages(OUT_PDF_BOX) as pdf:
    for r in REGULONS:
        panels = panels_for(r['name'])
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2),
                                 squeeze=False)
        for ax, p in zip(axes[0], panels):
            positions, groups = binned_groups(p['pc1'], p['score'], N_PC1_BINS)
            draw_boxbins(ax, positions, groups, p['color'],
                         f"{p['title']}\n{r['name']} regulon ({p['n']} targets)")
        fig.suptitle(f"{r['name']} regulon — mean-target expression by PC1 bin (boxplots)")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=DPI)
        plt.close(fig)
print(f'Saved {OUT_PDF_BOX}')
print('Done.')
