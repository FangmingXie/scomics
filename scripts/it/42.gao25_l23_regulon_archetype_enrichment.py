"""Fisher-exact enrichment of L2/3 archetype markers in SCENIC+ regulons (mouse gao25 V1).

Single-layer (L2/3) analog of scripts/it/41.regulon_archetype_enrichment.py, but the
regulon list is swapped to the gao25 V1 L2/3 eRegulon table. For each archetype, test
whether its marker gene set is over-represented among each regulon's target genes
(one-sided Fisher exact), then render the archetype x regulon heatmap.

Two gene-set resources are combined:
  - archetype markers: local_data/res/it/34.follow.two_L23_archetype_markers.tsv
                       (mouse IT L2/3 markers; cheng22 + yoo25, 3 archetypes)
  - regulon targets  : links/l23_evo/reguon_gene_table_gao25_v1l23.csv

Unlike script 41 (which reads pre-computed yoo25 regulon-target TSVs already in columns
regulon/TF/regulation_direction/Gene), the gao25 table is raw long-format (one row per
regulon-gene pair). The 4 columns this analysis consumes are derived inline (mirrors
scripts/l23_evo/35.human_l23_regulon_archetype_enrichment_mtg5.py):
  regulation_direction = TF2G_sign/R2G_sign ; regulon = TF_direction.
Only positive-R2G regulons ('+/+', '-/+') are kept; gao25 contains only these two.

Background universe: the L2/3 *tested expression gene set* the marker Wilcoxon ran over
(shared expressed / nonzero-variance genes across cheng22 + yoo25). It is not saved by
the marker scripts, so it is reconstructed here from the two h5ad inputs, replicating
scripts/it/33.follow.two_L4_archetype_scores.py:127-176 (minus the PCHA/marker steps).

Per (archetype a, regulon r) over universe U (|U| = N):
  x = |M_a & T_r|, M = |M_a|, T = |T_r|; 2x2 = [[x, M-x], [T-x, N-M-T+x]]
  p   = fisher_exact(2x2, alternative='greater')
  OR  = Haldane-Anscombe corrected odds ratio (+0.5 per cell); log2_or = log2(OR)
BH-FDR is applied across all (archetype, regulon) pairs.

Reads:
  local_data/res/it/34.follow.two_L23_archetype_markers.tsv
  local_data/res/it/34.harmony.two_L23_coords.tsv
  links/l23_evo/reguon_gene_table_gao25_v1l23.csv
  links/it/superdupermegaRNA_cheng22_IT_P28NR.h5ad
  links/it/superdupermegaRNA_yoo25_IT_P21.h5ad
Outputs:
  local_data/res/it/42.gao25_L2_3_regulon_archetype_enrichment.tsv   (long format)
  local_data/res/it/42.gao25_L2_3_enrichment_neglog10fdr.tsv         (matrix)
  local_data/res/it/42.gao25_L2_3_enrichment_log2or.tsv              (matrix)
  local_data/fig/it/42.gao25_L2_3_regulon_archetype_enrichment.html  (heatmap)
"""

import os

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import _write_fig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

INPUT_CHENG22 = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_cheng22_IT_P28NR.h5ad')
INPUT_YOO25 = os.path.join(PROJECT_ROOT, 'links', 'it', 'superdupermegaRNA_yoo25_IT_P21.h5ad')
REGULONS = os.path.join(PROJECT_ROOT, 'links', 'l23_evo', 'reguon_gene_table_gao25_v1l23.csv')

# shared dataset config (tag, h5ad path, subclass column, depth column)
DATASETS = [
    dict(tag='cheng22', path=INPUT_CHENG22, subclass_col='Subclass', depth_col='n_counts'),
    dict(tag='yoo25', path=INPUT_YOO25, subclass_col='Subclass', depth_col='total_counts'),
]

# single L2/3 config (markers / coords + subclass value + NOC); gao25 is V1 L2/3 only
LAYER = dict(layer='L2_3', subclass_val='L2/3', noc=3,
             markers='34.follow.two_L23_archetype_markers.tsv',
             coords='34.harmony.two_L23_coords.tsv',
             # rename L2/3 archetype labels: old A,B,C (archetype_1,2,3) -> C',B',A'
             arch_relabel={'archetype_1': "C'", 'archetype_2': "B'", 'archetype_3': "A'"})

KEEP_DIRECTIONS = {'+/+', '-/+'}  # keep only regulons with a positive R2G (second) sign
ARCHETYPE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']  # archetype_1 -> A, etc.
MIN_REGULON_GENES = 5    # drop regulons with fewer than this many targets in-universe
LOG2OR_SHOW = 2.0        # a regulon row is shown if log2 OR > this in >= 1 archetype
STAR_FDR = 1e-5          # heatmap cells marked '*' where FDR < this
COLOR_ABS = 5.0          # fixed log2 OR colorbar range [-COLOR_ABS, COLOR_ABS] for all heatmaps

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def reconstruct_gene_universe(coords_index, subclass_val, adatas):
    """Per-layer tested gene set: shared expressed/nonzero-var genes over the
    embedding cells. Replicates scripts/it/33.follow.two_L4_archetype_scores.py:127-176.

    `adatas` maps tag -> full AnnData (loaded once); this subsets to `subclass_val`.
    """
    subs = {}
    for d in DATASETS:
        a = adatas[d['tag']]
        a = a[a.obs[d['subclass_col']] == subclass_val].copy()
        subs[d['tag']] = a
        print(f"    {d['tag']:8s}: {a.n_obs} cells ({subclass_val}), {a.n_vars} genes")

    # shared gene set (inner join), ordered by cheng22, mito removed
    common_set = set(subs['cheng22'].var_names)
    for tag in [d['tag'] for d in DATASETS if d['tag'] != 'cheng22']:
        common_set &= set(subs[tag].var_names)
    common_genes = [g for g in subs['cheng22'].var_names if g in common_set]
    common_genes = [g for g in common_genes if not g.lower().startswith('mt-')]

    # CP10k -> log2 per dataset (validated depth + raw counts), then merge
    expr_frames = []
    for d in DATASETS:
        a = subs[d['tag']]
        depth = a.obs[d['depth_col']].values.astype(np.float64)
        assert np.all(np.isfinite(depth)) and np.all(depth > 0), \
            f"{d['tag']}: invalid depth column '{d['depth_col']}' (NaN or <=0)"
        xc = a.raw[:, common_genes].X
        xc = xc.toarray() if sp.issparse(xc) else np.asarray(xc, dtype=np.float64)
        xc = np.log2(xc / depth[:, None] * 1e4 + 1).astype(np.float32)
        idx = np.array([f"{d['tag']}:{n}" for n in a.obs_names])
        expr_frames.append(pd.DataFrame(xc, index=idx, columns=common_genes))

    expr_df = pd.concat(expr_frames)
    missing = np.setdiff1d(coords_index, expr_df.index.values)
    assert missing.size == 0, \
        f'{missing.size} coord cells missing from reconstructed expression, e.g. {missing[:5]}'
    expr_df = expr_df.reindex(coords_index)

    # expressed (nonzero total) + nonzero-variance
    X = expr_df.values
    keep = (X.sum(axis=0) > 0) & (X.var(axis=0) > 0)
    universe = expr_df.columns.values[keep]
    print(f'    gene universe: {keep.sum()} expressed nonzero-variance genes '
          f'(dropped {(~keep).sum()} of {keep.size})')
    return set(universe)


def log2_odds_ratio(x, M, T, N):
    """Haldane-Anscombe corrected log2 odds ratio for the 2x2 [[x,M-x],[T-x,N-M-T+x]]."""
    a, b, c, d = x + 0.5, (M - x) + 0.5, (T - x) + 0.5, (N - M - T + x) + 0.5
    return float(np.log2((a * d) / (b * c)))


def enrich_layer(cfg, adatas):
    layer, noc = cfg['layer'], cfg['noc']
    print(f'\n=== {layer} (gao25 V1 regulons) ===')

    coords = pd.read_csv(os.path.join(RES_DIR, cfg['coords']), sep='\t', index_col=0)
    universe = reconstruct_gene_universe(coords.index.values, cfg['subclass_val'], adatas)
    N = len(universe)

    # archetype marker sets (restricted to universe), ordered archetype_1..noc
    markers = pd.read_csv(os.path.join(RES_DIR, cfg['markers']), sep='\t')
    arch_labels = [f'archetype_{k + 1}' for k in range(noc)]
    M_sets = {a: set(markers.loc[markers['archetype'] == a, 'gene']) & universe for a in arch_labels}

    # gao25 regulon targets (raw long-format): derive the (regulon, TF,
    # regulation_direction, Gene) columns this analysis consumes.
    reg = pd.read_csv(REGULONS, sep='\t')  # tab-separated despite .csv extension
    reg['regulation_direction'] = reg['TF2G_sign'] + '/' + reg['R2G_sign']
    reg = reg[reg['regulation_direction'].isin(KEEP_DIRECTIONS)].copy()
    reg['regulon'] = reg['TF'] + '_' + reg['regulation_direction']
    reg = reg.drop_duplicates(subset=['regulon', 'Gene'])

    reg_meta = reg.drop_duplicates('regulon').set_index('regulon')[['TF', 'regulation_direction']]
    T_sets = {r: set(g['Gene']) & universe for r, g in reg.groupby('regulon')}
    n_total = len(T_sets)
    T_sets = {r: t for r, t in T_sets.items() if len(t) >= MIN_REGULON_GENES}
    print(f'    regulons: {len(T_sets)} kept (>= {MIN_REGULON_GENES} in-universe targets) '
          f'of {n_total}; universe N={N}')

    rows = []
    for r, T in T_sets.items():
        for a in arch_labels:
            M = M_sets[a]
            x = len(M & T)
            table = [[x, len(M) - x], [len(T) - x, N - len(M) - len(T) + x]]
            _, pval = scipy.stats.fisher_exact(table, alternative='greater')
            rows.append(dict(
                layer=layer, archetype=a, regulon=r,
                TF=reg_meta.loc[r, 'TF'], regulation_direction=reg_meta.loc[r, 'regulation_direction'],
                overlap=x, n_markers=len(M), n_targets=len(T), universe=N,
                log2_or=log2_odds_ratio(x, len(M), len(T), N), pval=pval,
            ))

    long = pd.DataFrame(rows)
    long['fdr'] = multipletests(long['pval'].values, method='fdr_bh')[1]
    long['neglog10_fdr'] = -np.log10(long['fdr'].clip(lower=np.nextafter(0, 1)))

    # archetype display labels: per-layer override (L2/3 A,B,C -> C',B',A')
    # else default archetype_1->A, archetype_2->B, ...
    col_label = cfg.get('arch_relabel') or {a: ARCHETYPE_LETTERS[k] for k, a in enumerate(arch_labels)}
    long['arch_letter'] = long['archetype'].map(col_label)

    out_long = os.path.join(RES_DIR, f'42.gao25_{layer}_regulon_archetype_enrichment.tsv')
    long.to_csv(out_long, sep='\t', index=False)
    print(f'    wrote -> {out_long} ({len(long)} pairs)')

    fdr_mat = long.pivot(index='regulon', columns='arch_letter', values='neglog10_fdr')
    or_mat = long.pivot(index='regulon', columns='arch_letter', values='log2_or')
    fdr_mat.to_csv(os.path.join(RES_DIR, f'42.gao25_{layer}_enrichment_neglog10fdr.tsv'), sep='\t')
    or_mat.to_csv(os.path.join(RES_DIR, f'42.gao25_{layer}_enrichment_log2or.tsv'), sep='\t')

    # ---- two heatmaps (+/+ and -/+), colored by log2 OR, '*' where FDR < STAR_FDR ----
    panels = [('+/+', 'activating'), ('-/+', 'repressing')]

    # collect each panel's shown matrices (regulons with log2 OR > LOG2OR_SHOW AND
    # FDR < STAR_FDR in >= 1 archetype within that sign)
    panel_show = []
    for sign, _lbl in panels:
        sub = long[long['regulation_direction'] == sign]
        orm = sub.pivot(index='regulon', columns='arch_letter', values='log2_or')
        fdm = sub.pivot(index='regulon', columns='arch_letter', values='fdr')
        keep = orm.index[((orm > LOG2OR_SHOW) & (fdm < STAR_FDR)).any(axis=1)]
        if len(keep) == 0:
            panel_show.append(None)
            continue
        # group rows by dominant archetype (peak log2 OR): A-block first, then B, C, ...
        # within a block, sort by descending peak log2 OR
        sub_or = orm.loc[keep]
        letter_rank = {c: i for i, c in enumerate(sorted(sub_or.columns))}
        ordering = pd.DataFrame({
            'prank': sub_or.idxmax(axis=1).map(letter_rank),
            'peak': sub_or.max(axis=1),
        })
        order = ordering.sort_values(['prank', 'peak'], ascending=[True, False]).index
        panel_show.append(dict(
            sign=sign, order=order,
            log2or=orm.loc[order],
            fdr=fdm.loc[order],
            overlap=sub.pivot(index='regulon', columns='arch_letter', values='overlap').loc[order],
            ntarg=sub.pivot(index='regulon', columns='arch_letter', values='n_targets').loc[order],
        ))
        print(f'    {sign}: {len(keep)} regulons '
              f'(log2 OR>{LOG2OR_SHOW} AND FDR<{STAR_FDR:g} in >=1 archetype) shown')

    if all(p is None for p in panel_show):
        print(f'    [skip heatmap] no regulons meeting both criteria for {layer}')
        return long

    max_rows = max(len(p['order']) for p in panel_show if p is not None)

    # one independent color axis per panel; place its colorbar beside its subplot
    cax_names = ['coloraxis', 'coloraxis2']
    cbar_x = [0.46, 1.0]  # colorbar x positions (subplot1 gap, subplot2 right edge)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.24,
        subplot_titles=[f'{lbl} ({sign})  n={len(p["order"]) if p else 0}'
                        for (sign, lbl), p in zip(panels, panel_show)],
    )
    layout_caxes = {}
    for col, p in enumerate(panel_show, start=1):
        if p is None:
            continue
        cax = cax_names[col - 1]
        stars = np.where(p['fdr'].values < STAR_FDR, '*', '')
        customdata = np.dstack([p['overlap'].values, p['ntarg'].values, p['fdr'].values])
        fig.add_trace(go.Heatmap(
            z=p['log2or'].values, x=list(p['log2or'].columns), y=list(p['log2or'].index),
            coloraxis=cax, text=stars, texttemplate='%{text}',
            textfont=dict(size=14, color='black'),
            customdata=customdata,
            hovertemplate=('regulon=%{y}<br>archetype=%{x}<br>log2 OR=%{z:.2f}'
                           '<br>overlap=%{customdata[0]}<br>n_targets=%{customdata[1]}'
                           '<br>FDR=%{customdata[2]:.2e}<extra></extra>'),
        ), row=1, col=col)
        fig.update_yaxes(autorange='reversed', row=1, col=col)
        fig.update_xaxes(title_text='archetype', row=1, col=col)
        layout_caxes[cax] = dict(
            colorscale='RdBu_r', cmid=0, cmin=-COLOR_ABS, cmax=COLOR_ABS,
            colorbar=dict(title=f'log2 OR<br>({p["sign"]})', x=cbar_x[col - 1],
                          xanchor='left', len=0.9, thickness=14),
        )

    fig.update_layout(
        title=f'gao25 V1 L2/3 IT — archetype marker enrichment in regulons '
              f'(log2 OR; * FDR<{STAR_FDR:g}; N={N} genes)',
        height=max(400, 18 * max_rows + 180), width=1000,
        **layout_caxes,
    )
    out_html = os.path.join(FIG_DIR, f'42.gao25_{layer}_regulon_archetype_enrichment.html')
    _write_fig(fig, out_html)  # HTML whose screenshot button exports SVG by default
    return long


def main():
    print('Loading h5ad inputs once...')
    adatas = {d['tag']: ad.read_h5ad(d['path']) for d in DATASETS}
    for d in DATASETS:
        print(f"  {d['tag']:8s}: {adatas[d['tag']].n_obs} cells, {adatas[d['tag']].n_vars} genes")

    enrich_layer(LAYER, adatas)


if __name__ == '__main__':
    main()
