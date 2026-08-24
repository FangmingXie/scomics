"""58's subclass-overlap page for all eight IEG regulons, plus a summary (multi-page PDF).

58 asks of Fosb whether "the Fosb regulon" is one gene set the four subclasses agree on or
four sets sharing a name. That question is not about Fosb -- it applies to every row of 54f's
dot matrix, and its answer is the caveat those figures depend on. This runs 58 over 54f's
eight IEG TFs and collects the result.

Pages: one 58 page per TF in IEG_TFS order, then a summary comparing them:
  A  union composition -- for each TF, the union of its target sets split by how many
     subclasses called each gene. Degree is only comparable between TFs with the same
     number of catalogues, so the count of catalogues is printed on every row.
  B  pairwise agreement -- TF x subclass-pair, colour = log2 enrichment over the
     shared-testable null, area = Jaccard, black outline = FDR < 58.STAR_FDR. A pair is
     blank where either subclass has no regulon for that TF; that absence is the point of
     keeping all eight TFs as rows, exactly as 54f keeps its uncalled regulons as rows.

Fos, Fosl2, Egr2 and Egr4 exist in fewer than four catalogues and Egr2 in only one -- see
`load_sets`'s printout. A single-catalogue TF has no overlap to measure, so its page carries
panel C alone and it contributes no row to the summary's panel B; it is still drawn and
still listed, because "SCENIC+ called this regulon in one subclass out of four" is a result.

Reads (via 58):
  local_data/res/it/40.yoo25_<layer>_regulon_targets.tsv
Outputs:
  local_data/res/it/58b.ieg_regulon_target_membership.tsv   (TF x gene x subclass, long)
  local_data/res/it/58b.ieg_regulon_pairwise_overlap.tsv    (all TFs, long)
  local_data/res/it/58b.ieg_regulon_union_composition.tsv   (per TF, genes by degree)
  local_data/fig/it/58b.ieg_regulon_target_overlap.pdf      (8 TF pages + summary)
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

PARENT_SCRIPT = os.path.join(SCRIPTS_DIR, 'it', '58.fosb_regulon_target_overlap.py')
OUT_MEMBERSHIP = os.path.join(RES_DIR, '58b.ieg_regulon_target_membership.tsv')
OUT_PAIRWISE = os.path.join(RES_DIR, '58b.ieg_regulon_pairwise_overlap.tsv')
OUT_COMPOSITION = os.path.join(RES_DIR, '58b.ieg_regulon_union_composition.tsv')
OUT_PDF = os.path.join(FIG_DIR, '58b.ieg_regulon_target_overlap.pdf')

# 54f.IEG_TFS: the AP-1 arm then the EGR arm, in that order rather than sorted so the two
# families read as blocks. Not imported -- 54f pins the row set of ITS figure, this pins the
# page order of this one, and the two should be free to move apart.
IEG_TFS = ['Fos', 'Fosb', 'Fosl2', 'Junb', 'Egr1', 'Egr2', 'Egr3', 'Egr4']

# summary panel A: degree 1..4 on a light-to-dark ramp, so "shared by more subclasses" reads
# as darker without implying the degrees are a continuous quantity
DEGREE_COLOR = ['#e3e3e3', '#b0c4de', '#5c85c0', '#1f3f70']
# summary panel B area encoding, as in 54d: a dot at JACC_REF covers SIZE_REF points^2 and
# area scales linearly with the Jaccard index, so twice the area reads as twice the agreement
# JACC_REF sits just above the observed maximum over all eight TFs (0.382, Egr1 L4 / L6IT)
JACC_REF, SIZE_REF = 0.40, 230.0
SIZE_LEGEND = [0.05, 0.15, 0.25, 0.35]
BOX_LW = 1.4

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_composition(ax, comp, m58):
    """Summary panel A: each TF's union split by the number of subclasses that called a gene."""
    y = np.arange(len(comp))
    left = np.zeros(len(comp))
    max_deg = int(comp['n_subclasses'].max())
    for d in range(1, max_deg + 1):
        n = comp[f'degree_{d}'].values
        ax.barh(y, n, left=left, height=0.62, color=DEGREE_COLOR[d - 1], zorder=3,
                label=f'{d} subclass' + ('' if d == 1 else 'es'))
        for yi, (l, v) in enumerate(zip(left, n)):
            if v >= 0.06 * comp['union'].max():
                ax.text(l + v / 2, yi, str(int(v)), ha='center', va='center', fontsize=6.5,
                        color='white' if d >= 3 else '0.25', zorder=4)
        left = left + n
    for yi, row in enumerate(comp.itertuples()):
        ax.text(row.union + 0.012 * comp['union'].max(), yi,
                f'{row.union}  ({row.n_subclasses} of 4 catalogues)',
                va='center', ha='left', fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(comp['TF'], fontsize=8)
    ax.set_ylim(len(comp) - 0.4, -0.6)
    ax.set_xlim(0, comp['union'].max() * 1.33)
    ax.set_xlabel('genes in the union of the target sets', fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    ax.grid(True, axis='x', color='0.90', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, title='called by', title_fontsize=7, frameon=False,
              loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=4, handlelength=1.2)


def draw_agreement(ax, cax, pw, m58):
    """Summary panel B: TF x subclass-pair dots, colour = enrichment, area = Jaccard."""
    labels = [lbl for _l, lbl in m58.LAYER_LABEL]
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
    col = {p: i for i, p in enumerate(pairs)}
    row = {tf: i for i, tf in enumerate(IEG_TFS)}

    assert pw['jaccard'].max() <= JACC_REF + 1e-9, \
        f'Jaccard {pw["jaccard"].max():.3f} exceeds the size reference {JACC_REF}'
    norm = Normalize(vmin=-m58.COLOR_ABS, vmax=m58.COLOR_ABS)
    xs = [col[(r.subclass_a, r.subclass_b)] for r in pw.itertuples()]
    ys = [row[r.TF] for r in pw.itertuples()]
    sig = pw['fdr'] < m58.STAR_FDR
    sc = ax.scatter(xs, ys, s=pw['jaccard'] / JACC_REF * SIZE_REF, c=pw['log2_enr'],
                    cmap='RdBu_r', norm=norm, zorder=3,
                    edgecolors=np.where(sig, 'black', 'none'),
                    linewidths=np.where(sig, BOX_LW, 0.0))

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f'{a} / {b}' for a, b in pairs], rotation=45, ha='right', fontsize=7.5)
    ax.set_yticks(range(len(IEG_TFS)))
    ax.set_yticklabels(IEG_TFS, fontsize=8)
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(len(IEG_TFS) - 0.4, -0.6)
    ax.grid(True, color='0.90', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')

    cb = plt.colorbar(sc, cax=cax)
    cb.set_label('log2 obs/exp over genes\ntestable in both', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker='o', ls='none', markerfacecolor='0.55',
                      markeredgecolor='none', markersize=np.sqrt(j / JACC_REF * SIZE_REF),
                      label=f'{j:.2f}') for j in SIZE_LEGEND]
    handles.append(Line2D([], [], marker='o', ls='none', markerfacecolor='white',
                          markeredgecolor='black', markeredgewidth=BOX_LW, markersize=8,
                          label=f'FDR < {m58.STAR_FDR:g}'))
    ax.legend(handles=handles, fontsize=7, title='Jaccard', title_fontsize=7, frameon=False,
              loc='upper left', bbox_to_anchor=(1.14, 1.02), labelspacing=1.1,
              handletextpad=1.0, borderpad=0.6)


def summary_figure(comp, pw, m58):
    fig = plt.figure(figsize=(9.2, 8.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.0, 0.03, 0.30],
                  hspace=0.42, wspace=0.06)
    draw_composition(fig.add_subplot(gs[0, 0]), comp, m58)
    fig.add_subplot(gs[0, 1]).axis('off')
    fig.add_subplot(gs[0, 2]).axis('off')
    draw_agreement(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), pw, m58)
    fig.add_subplot(gs[1, 2]).axis('off')

    for x, y, s in [(0.02, 0.965, 'A  how much of each IEG regulon is shared'),
                    (0.02, 0.475, 'B  agreement between catalogues, per TF and pair')]:
        fig.text(x, y, s, fontsize=10, fontweight='bold')
    fig.suptitle('The eight IEG regulons across the four IT subclass catalogues (yoo25)',
                 fontsize=9, y=1.005)
    return fig


def main():
    m58 = load_module(PARENT_SCRIPT, 'script58')

    pages, pw_all, mem_all, comp_rows = [], [], [], []
    for tf in IEG_TFS:
        table, pw, fig = m58.main(tf, save=False)
        pages.append(fig)
        pw_all.append(pw)
        mem_all.append(table.assign(TF=tf))
        genes = table.drop_duplicates('gene')
        n_sub = int(table['subclass'].nunique())
        rec = dict(TF=tf, n_subclasses=n_sub, union=int(genes['gene'].nunique()),
                   core=int((genes['degree'] == n_sub).sum()),
                   private=int((genes['degree'] == 1).sum()))
        rec.update({f'degree_{d}': int((genes['degree'] == d).sum()) for d in range(1, 5)})
        comp_rows.append(rec)
        print()

    comp = pd.DataFrame(comp_rows)
    pw_all = pd.concat(pw_all, ignore_index=True)
    mem_all = pd.concat(mem_all, ignore_index=True)[['TF', 'gene', 'subclass', 'degree',
                                                     'pattern']]
    comp.to_csv(OUT_COMPOSITION, sep='\t', index=False)
    pw_all.to_csv(OUT_PAIRWISE, sep='\t', index=False)
    mem_all.to_csv(OUT_MEMBERSHIP, sep='\t', index=False)

    print('summary (union / called by every catalogue that has the TF / called by one):')
    for r in comp.itertuples():
        print(f'  {r.TF:<6} {r.n_subclasses} catalogues, union {r.union:>4}, '
              f'core {r.core:>3} ({100 * r.core / r.union:>3.0f}%), '
              f'private {r.private:>4} ({100 * r.private / r.union:>3.0f}%)')
    sig = pw_all['fdr'] < m58.STAR_FDR
    print(f'  {int(sig.sum())}/{len(pw_all)} subclass pairs overlap above the shared-testable '
          f'null at FDR < {m58.STAR_FDR:g}; Jaccard {pw_all["jaccard"].min():.3f}-'
          f'{pw_all["jaccard"].max():.3f}')

    with PdfPages(OUT_PDF) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        fig = summary_figure(comp, pw_all, m58)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    for path in (OUT_PDF, OUT_MEMBERSHIP, OUT_COMPOSITION, OUT_PAIRWISE):
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
