"""How much do the archetype-differentiating genes overlap across IT subclasses? (UpSet)

Scripts 33-36 each fit archetypes within ONE subclass and write the genes that differentiate
them (`{34,33,35,36}.follow.two_{L23,L4,L5IT,L6IT}_archetype_markers.tsv`). Those four marker
sets are computed independently, so nothing forces them to agree -- this figure asks how far
they do: pool each subclass's archetype markers into a single per-subclass gene set, then count
the genes in every observed combination of subclasses.

The layout is a standard UpSet plot, which is the honest form for 4 sets (a 4-way Venn hides
sizes and there are 15 possible regions):
  - top:   one bar per observed subclass combination, height = number of genes in EXACTLY that
           combination (the regions are disjoint, so the bars sum to the total gene count).
           Bars are sorted large-to-small and shaded by degree (how many subclasses share the
           gene), light = subclass-specific, dark = shared by all four.
  - middle:the dot matrix naming each bar's combination.
  - left:  each subclass's total pooled marker count.
  - right: the coarsest split -- one bar of genes unique to each subclass, plus a single bar
           pooling everything shared by two or more. Five bars covering all the genes.

Pooling is across archetypes ON PURPOSE: this asks "do these layers differentiate their
archetypes with the same genes at all", not "is L2/3 A' the same program as L4 A'". A gene
counted here may well mark a different archetype in each subclass. Within a subclass no gene
marks two archetypes (asserted), so the pooled set is a plain union.

Reads:
  local_data/res/it/{34,33,35,36}.follow.two_{L23,L4,L5IT,L6IT}_archetype_markers.tsv
Outputs:
  local_data/fig/it/60.it_archetype_marker_overlap_upset.pdf
  local_data/res/it/60.it_archetype_marker_overlap.tsv   (gene -> combination, degree)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR  = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
OUT_PDF  = os.path.join(FIG_DIR, '60.it_archetype_marker_overlap_upset.pdf')
OUT_TSV  = os.path.join(RES_DIR, '60.it_archetype_marker_overlap.tsv')

# Laminar order, top to bottom in the matrix.
SUBCLASSES = [
    dict(subclass='L2/3', markers='34.follow.two_L23_archetype_markers.tsv'),
    dict(subclass='L4',   markers='33.follow.two_L4_archetype_markers.tsv'),
    dict(subclass='L5IT', markers='35.follow.two_L5IT_archetype_markers.tsv'),
    dict(subclass='L6IT', markers='36.follow.two_L6IT_archetype_markers.tsv'),
]

# --- parameters ---
DEGREE_SHADES = ['#c9c9c9', '#9e9e9e', '#6b6b6b', '#3a3a3a']   # degree 1..4, light -> dark
SET_BAR_COLOR = '#7f8fa6'      # per-subclass total bars
SHARED_COLOR  = '#5a5a5a'      # the pooled "shared by >= 2 subclasses" bar
DOT_ON        = '#3a3a3a'
DOT_OFF       = '#e0e0e0'
DOT_SIZE      = 70
BAR_LABEL_FS  = 7
FIG_W, FIG_H  = 12.5, 6.0
DPI           = 300


def load_sets():
    """{subclass: set(genes)} -- each subclass's archetype markers pooled across archetypes."""
    sets, per_arch = {}, {}
    for cfg in SUBCLASSES:
        path = os.path.join(RES_DIR, cfg['markers'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{cfg['subclass']}: missing {path}")
        mk = pd.read_csv(path, sep='\t')
        dup = mk['gene'].duplicated().sum()
        if dup:
            raise ValueError(f'{path}: {dup} gene(s) mark more than one archetype; the pooled '
                             f'set would not be a plain union')
        sets[cfg['subclass']] = set(mk['gene'])
        per_arch[cfg['subclass']] = mk.groupby('archetype').size().to_dict()
    return sets, per_arch


def intersections(sets):
    """Genes -> their exact subclass combination, as a size-sorted table of disjoint regions."""
    names = [cfg['subclass'] for cfg in SUBCLASSES]
    rows = []
    for gene in sorted(set().union(*sets.values())):
        combo = tuple(n for n in names if gene in sets[n])
        rows.append({'gene': gene, 'combination': '+'.join(combo), 'degree': len(combo)})
    genes = pd.DataFrame(rows)
    combos = (genes.groupby(['combination', 'degree']).size().reset_index(name='n_genes')
              .sort_values(['n_genes', 'degree'], ascending=[False, True])
              .reset_index(drop=True))
    return genes, combos


def draw(sets, combos, genes):
    names = [cfg['subclass'] for cfg in SUBCLASSES]
    members = [set(c.split('+')) for c in combos['combination']]
    x = np.arange(len(combos))

    plt.rcParams['pdf.fonttype'] = 42       # editable vector text
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 4.2, 1.4], height_ratios=[3, 1.5],
                          wspace=0.14, hspace=0.06)   # wspace keeps the subclass names clear
                                                       # of the first matrix column
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)
    ax_uni = fig.add_subplot(gs[0, 2])

    # --- intersection sizes (disjoint regions, so these sum to the total gene count) ---
    bars = ax_bar.bar(x, combos['n_genes'],
                      color=[DEGREE_SHADES[d - 1] for d in combos['degree']],
                      edgecolor='0.3', linewidth=0.5)
    ax_bar.bar_label(bars, fontsize=BAR_LABEL_FS, padding=1)
    ax_bar.set_ylabel('genes in exactly\nthis combination')
    ax_bar.tick_params(labelbottom=False)
    ax_bar.set_ylim(0, combos['n_genes'].max() * 1.12)
    sns.despine(ax=ax_bar)

    # --- dot matrix ---
    for i in range(len(names)):                       # alternating row bands for readability
        if i % 2 == 0:
            ax_mat.axhspan(i - 0.5, i + 0.5, color='0.96', zorder=0)
    for j, mem in enumerate(members):
        on = [i for i, n in enumerate(names) if n in mem]
        ax_mat.plot([j, j], [min(on), max(on)], color=DOT_ON, lw=1.6, zorder=1)
        ax_mat.scatter([j] * len(names), range(len(names)), s=DOT_SIZE, zorder=2,
                       color=[DOT_ON if n in mem else DOT_OFF for n in names])
    ax_mat.set_xlim(-0.6, len(combos) - 0.4)
    ax_mat.set_ylim(len(names) - 0.5, -0.5)
    ax_mat.set_yticks(range(len(names)))
    ax_mat.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for side in ('top', 'right', 'bottom', 'left'):
        ax_mat.spines[side].set_visible(False)

    # --- per-subclass totals (mirrored, so the bars grow toward the matrix) ---
    totals = [len(sets[n]) for n in names]
    ax_set.barh(range(len(names)), totals, color=SET_BAR_COLOR, edgecolor='0.3', linewidth=0.5)
    for i, t in enumerate(totals):
        # x is inverted, so 'right' puts the label just PAST the bar tip, clear of the bar.
        ax_set.text(t + max(totals) * 0.03, i, str(t), va='center', ha='right',
                    fontsize=BAR_LABEL_FS)
    ax_set.set_xlim(max(totals) * 1.35, 0)
    ax_set.set_yticks(range(len(names)))
    ax_set.set_yticklabels(names, fontsize=9)
    ax_set.yaxis.tick_right()
    ax_set.tick_params(right=False)
    ax_set.set_xlabel('markers\nper subclass', fontsize=8)
    sns.despine(ax=ax_set, left=True, right=True, top=True)

    # --- coarsest split: unique to each subclass, plus everything shared by >= 2 ---
    n_sets = len(names)
    uniq = [int(((genes['degree'] == 1) & (genes['combination'] == n)).sum()) for n in names]
    shared = int((genes['degree'] >= 2).sum())
    ubars = ax_uni.bar(range(n_sets + 1), uniq + [shared],
                       color=[DEGREE_SHADES[0]] * n_sets + [SHARED_COLOR],
                       edgecolor='0.3', linewidth=0.5)
    ax_uni.bar_label(ubars, fontsize=BAR_LABEL_FS, padding=1)
    ax_uni.set_xticks(range(n_sets + 1))
    ax_uni.set_xticklabels(names + [f'shared\n(>= 2)'], fontsize=8, rotation=45, ha='right')
    ax_uni.set_ylabel('genes', fontsize=9)
    ax_uni.set_ylim(0, max(uniq + [shared]) * 1.12)
    ax_uni.tick_params(labelsize=8)
    sns.despine(ax=ax_uni)

    fig.suptitle('Archetype-differentiating genes: overlap across IT subclasses\n'
                 '(two-dataset cheng22+yoo25 fits; each subclass pooled across its archetypes)',
                 fontsize=11)
    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
    plt.close(fig)


print('--- IT archetype-marker overlap across subclasses (UpSet) ---')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

sets, per_arch = load_sets()
for cfg in SUBCLASSES:
    s = cfg['subclass']
    print(f'  {s:5s} {len(sets[s]):4d} markers  ' +
          '  '.join(f'{a}={n}' for a, n in sorted(per_arch[s].items())))

genes, combos = intersections(sets)
print(f'  {len(genes)} distinct genes in {len(combos)} observed combinations')
by_degree = genes.groupby('degree').size()
for d, n in by_degree.items():
    print(f'    shared by {d} subclass(es): {n:4d} genes')
top = combos.head(6).itertuples()
print('  largest combinations: ' + '  | '.join(f'{r.combination} {r.n_genes}' for r in top))
all_four = genes.loc[genes['degree'] == len(SUBCLASSES), 'gene'].tolist()
print(f'  in all {len(SUBCLASSES)} subclasses ({len(all_four)}): {", ".join(all_four)}')

genes.to_csv(OUT_TSV, sep='\t', index=False)
print(f'  Saved {OUT_TSV}')
draw(sets, combos, genes)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
