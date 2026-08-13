"""Mouse IT archetype programs over human developmental stage (Wang25 10x multiome GEX, V1).

Adapted from scripts/l23_evo/50: same per-stage activity readout, but the gene sets are the
mouse IT archetype markers defined in scripts/it (33/34/35/36.follow.*) instead of SCENIC+
regulon targets, mapped to human orthologs as in scripts/l23_evo/60.

  - Panel 1 (L2/3):  archetypes A, B, C   scored on EN-L2_3-IT cells
  - Panel 2 (L4):    archetypes A, B, C   scored on EN-L4-IT cells
  - Panel 3 (L5 IT): archetypes A, B      scored on EN-L5-IT cells
  - Panel 4 (L6 IT): archetypes A, B, C   scored on EN-L6-IT cells

Letters map positionally onto archetype_1..k in the marker tables — no relabeling. (The
ARCH_RELABEL {A:"C'", B:"B'", C:"A'"} used in scripts/it/41,48,50 is a display convention for
the L2/3 regulon figures only; plain A/B/C is used here so all four panels are consistent.)

Marker genes are kept only if their human ortholog is also a human HVG for that layer — the
2000-gene list behind local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv (adult human
IT data, script 02), matched panel-to-panel by layer token. This restricts each archetype to
the part of its program that is variable in human, i.e. the same gene universe the
cross-species axis matching runs on.

Per-cell score (script-50 method): each kept gene's log2(CP10k+1) is max-scaled to [0, 1]
across cells (divided by the NORM_MAX_PCTILE percentile, no baseline subtraction, so no
expression stays 0) then averaged across the archetype's genes. Normalization is computed
within each panel's own cell subset, so panels are self-contained.

Cells are pooled by developmental stage (second trimester -> third trimester -> infancy ->
adolescence), V1 only. One figure, 1 row x 4 columns: each archetype's per-stage mean +/- SEM
is max-scaled across stages (divided by its largest per-stage mean), so 1 marks the peak stage
and 0 marks no activity; relative magnitudes are preserved. The raw per-stage means are still
written to the output TSV.

Reads:
  links/it_evo/wang25_human_multiome_gex_it.h5ad
  local_data/res/it/{34,33,35,36}.follow.two_{L23,L4,L5IT,L6IT}_archetype_markers.tsv
  local_data/res/it_evo/02.human_{L23,L4,L5IT,L6IT}_varimax_loadings.tsv   (human HVG list)
  data/human_mouse_orthologs.tsv
Outputs:
  local_data/fig/it_evo/25.wang25_multiome_it_mouse_archetype_expr_over_stage.pdf
  local_data/res/it_evo/25.wang25_multiome_it_mouse_archetype_expr_over_stage.tsv
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42     # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_H5AD      = os.path.join(PROJECT_ROOT, 'links', 'it_evo', 'wang25_human_multiome_gex_it.h5ad')
IN_MARKER_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IN_HVG_TMPL  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo',
                            '02.human_{token}_varimax_loadings.tsv')
IN_ORTHOLOGS = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')

OUT_PDF = os.path.join(OUT_FIG_DIR, '25.wang25_multiome_it_mouse_archetype_expr_over_stage.pdf')
OUT_TSV = os.path.join(OUT_RES_DIR, '25.wang25_multiome_it_mouse_archetype_expr_over_stage.tsv')

# Panels: (label, human Type, layer token for the human HVG list, mouse marker TSV basename,
# archetype letters).
PANELS = [
    ('L2/3',  'EN-L2_3-IT', 'L23',  '34.follow.two_L23_archetype_markers.tsv',  ['A', 'B', 'C']),
    ('L4',    'EN-L4-IT',   'L4',   '33.follow.two_L4_archetype_markers.tsv',   ['A', 'B', 'C']),
    ('L5 IT', 'EN-L5-IT',   'L5IT', '35.follow.two_L5IT_archetype_markers.tsv', ['A', 'B']),
    ('L6 IT', 'EN-L6-IT',   'L6IT', '36.follow.two_L6IT_archetype_markers.tsv', ['A', 'B', 'C']),
]
# Per-line colors/markers (one per archetype letter).
LINE_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINE_MARKERS = ['o', 's', '^', 'D', 'v']

KEEP_REGION = 'V1'             # matches script 50 (V1 only)
TYPE_COL    = 'Type'
REGION_COL  = 'Region'
GROUP_COL   = 'Group'
CP_TARGET   = 1e4              # CP10k
NORM_MAX_PCTILE = 99           # per-gene max-scaling percentile (maps to 1), before averaging

# Developmental stages in temporal order (obs values -> display labels).
STAGE_ORDER  = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']
STAGE_LABELS = {'Second_trimester': 'Second trimester',
                'Third_trimester': 'Third trimester',
                'Infancy': 'Infancy',
                'Adolescence': 'Adolescence'}


def load_ortholog_map():
    """1-to-1 mouse->human symbol map (same construction as script 60)."""
    ortho = pd.read_csv(IN_ORTHOLOGS, sep='\t')
    ortho = ortho.drop_duplicates('human_symbol').drop_duplicates('mouse_symbol')
    return dict(zip(ortho['mouse_symbol'], ortho['human_symbol']))


def archetype_stage_stats(genes, X, depth, stage, stages, name2idx):
    """Per-stage mean+/-SEM (and stage-level max-scaling) of an archetype's marker-gene activity."""
    # Max-scale each gene across cells (NORM_MAX_PCTILE -> 1, 0 stays 0), then average.
    # No baseline subtraction, so absent expression maps to 0.
    cols = [name2idx[g] for g in genes]
    l2 = np.log2(X[:, cols] / depth[:, None] * CP_TARGET + 1.0)   # (n_cells, n_genes)
    hi = np.percentile(l2, NORM_MAX_PCTILE, axis=0)
    hi_g = np.where(hi > 0, hi, 1.0)
    norm = np.clip(l2 / hi_g, 0.0, 1.0)
    score = norm.mean(axis=1)

    df = pd.DataFrame({'stage': stage, 'score': score})
    grp = df.groupby('stage', observed=True)['score']
    stats = pd.DataFrame({
        'n_cells': grp.size(),
        'mean': grp.mean(),
        'std': grp.std(ddof=1),
    }).reindex(stages).reset_index().rename(columns={'index': 'stage'})
    stats['sem'] = stats['std'] / np.sqrt(stats['n_cells'])

    mmax = stats['mean'].max()
    if mmax <= 0:
        raise ValueError(f"Max per-stage mean is {mmax}; cannot max-scale")
    stats['mean_norm'] = stats['mean'] / mmax
    stats['sem_norm'] = stats['sem'] / mmax
    return stats


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_RES_DIR, exist_ok=True)

    mouse_to_human = load_ortholog_map()
    print(f'{len(mouse_to_human)} 1-to-1 mouse->human orthologs')

    print(f'Loading {IN_H5AD}')
    adata = ad.read_h5ad(IN_H5AD)
    human_genes = set(adata.var_names)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.9 * len(PANELS), 3.9), squeeze=False)
    all_stats = []

    for col, (label, human_type, token, marker_file, letters) in enumerate(PANELS):
        # --- cells: this IT type, V1 only ---
        sub = adata[((adata.obs[TYPE_COL] == human_type)
                     & (adata.obs[REGION_COL] == KEEP_REGION)).values].copy()
        if sub.n_obs == 0:
            raise ValueError(f"No {human_type} cells in region {KEEP_REGION}")

        stage = sub.obs[GROUP_COL].astype(str).values
        unknown = set(stage) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown developmental stage(s) not in STAGE_ORDER: {unknown}")
        stages = [s for s in STAGE_ORDER if s in set(stage)]
        print(f'[{label}] {sub.n_obs} {human_type} cells ({KEEP_REGION}) across '
              f'{len(stages)} stages: {stages}')

        X = sub.X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = X.astype(np.float64)
        depth = X.sum(axis=1)
        if np.any(depth == 0):
            raise ValueError(f"Found {human_type} cells with zero total counts; cannot normalize")
        name2idx = {g: i for i, g in enumerate(sub.var_names)}

        # --- mouse markers -> human orthologs present in the data and in this layer's HVGs ---
        markers = pd.read_csv(os.path.join(IN_MARKER_DIR, marker_file), sep='\t')
        hvg = set(pd.read_csv(IN_HVG_TMPL.format(token=token), sep='\t', index_col=0).index)
        keep = human_genes & hvg
        print(f'  {len(hvg)} human {token} HVGs, {len(keep)} of them in the multiome data')

        xpos = np.arange(len(stages))
        ax_norm = axes[0, col]
        n_by_stage = None

        for i, letter in enumerate(letters):
            arch_id = f'archetype_{i + 1}'
            mouse_genes = markers.loc[markers['archetype'] == arch_id, 'gene'].tolist()
            if not mouse_genes:
                raise ValueError(f"No markers for {arch_id} in {marker_file}")
            used = [mouse_to_human[g] for g in mouse_genes
                    if g in mouse_to_human and mouse_to_human[g] in keep]
            if not used:
                raise ValueError(f"No ortholog-mapped human-HVG genes found in the data for "
                                 f"{label} {letter} ({arch_id}, {len(mouse_genes)} mouse markers)")
            print(f'  {letter} ({arch_id}): {len(used)}/{len(mouse_genes)} mouse markers '
                  f'mapped to human HVGs')

            stats = archetype_stage_stats(used, X, depth, stage, stages, name2idx)
            n_by_stage = stats['n_cells'].values
            stats.insert(0, 'panel', label)
            stats.insert(1, 'human_type', human_type)
            stats.insert(2, 'archetype', letter)
            stats.insert(3, 'arch_id', arch_id)
            stats.insert(4, 'n_markers_mouse', len(mouse_genes))
            stats.insert(5, 'n_genes_used', len(used))
            all_stats.append(stats)

            ax_norm.errorbar(xpos, stats['mean_norm'].values, yerr=stats['sem_norm'].values,
                             fmt='-' + LINE_MARKERS[i % len(LINE_MARKERS)],
                             color=LINE_COLORS[i % len(LINE_COLORS)], markersize=5, capsize=2.5,
                             linewidth=1.5, elinewidth=1.0, label=f'{letter} ({len(used)})')

        ticklabels = [f'{STAGE_LABELS.get(s, s)}\n(n={n})' for s, n in zip(stages, n_by_stage)]
        ax_norm.set_xlim(-0.4, len(stages) - 0.6)
        ax_norm.set_xticks(xpos)
        ax_norm.set_xticklabels(ticklabels, rotation=45, ha='right', fontsize=8)
        ax_norm.set_xlabel('Developmental stage')
        ax_norm.set_title(f'{label} — {human_type}', fontsize=11)
        ax_norm.legend(fontsize=8, frameon=False, title='archetype (n genes)', title_fontsize=8)
        ax_norm.spines['top'].set_visible(False)
        ax_norm.spines['right'].set_visible(False)
        if col == 0:
            ax_norm.set_ylabel('mouse archetype score\n(max-scaled across stages, peak = 1)',
                               fontsize=8)

    # [0,1] by construction — share limits across panels.
    for ax in axes[0]:
        ax.set_ylim(0, 1.12)

    fig.suptitle('Mouse IT archetype programs over human developmental stage — '
                 f'Wang25 multiome ({KEEP_REGION})', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    out = pd.concat(all_stats, ignore_index=True)
    out.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_TSV} ({len(out)} rows)')


if __name__ == '__main__':
    main()
