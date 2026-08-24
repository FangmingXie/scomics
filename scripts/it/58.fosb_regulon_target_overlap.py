"""How much of the Fosb regulon is shared across the four IT subclasses (PDF).

Everything in the 54/55 family tests a regulon's targets against archetype markers and never
asks the prior question: is "the Fosb regulon" one gene set that all four subclasses agree on,
or four different sets that happen to carry the same TF name? SCENIC+ was run per subclass
(40, one catalogue per L2/3, L4, L5IT, L6IT), so Fosb_+/+ is inferred independently four
times, and the row labelled Fosb in 54f's dot matrix is a different gene set in each of its
four column blocks. This script draws that: how the union of the target sets partitions into
subclass-private, partly shared and fully shared genes.

Three panels, one question each:
  A  UpSet -- every non-empty membership pattern over the subclasses that have the regulon,
     as an exclusive count (a gene is in exactly one bar), with each subclass's total.
  B  pairwise overlap -- observed shared targets, coloured by log2 enrichment over a
     hypergeometric null and starred at FDR < STAR_FDR. The null is drawn from the genes
     TESTABLE in both subclasses (targeted by some activating regulon in each catalogue),
     not from all genes: a gene absent from one subclass's SCENIC+ output could never have
     been called a target there, and scoring it as a miss would understate every overlap.
  C  the fully shared core -- the genes called in every subclass that has the regulon.

Only activating (+/+) regulons are used, matching 41b.SIGN and the whole 54/55 line, and the
targets are 40's direct-over-extended table verbatim: no re-derivation happens here, so the
gene sets are exactly the ones the enrichment scripts consumed.

The TF is a parameter (`main(tf)`), not a hard-wired name, because the same question applies
to all eight IEG regulons -- 58b is that entry point and imports this module. A TF the
catalogue called in only ONE subclass has no overlap to draw and is reported rather than
plotted; that is a fact about the catalogue, so it is printed loudly and 58b gives it a page
saying so instead of dropping it from the deck.

Reads:
  local_data/res/it/40.yoo25_<layer>_regulon_targets.tsv
Outputs:
  local_data/res/it/58.<tf>_regulon_target_membership.tsv   (gene x subclass, long)
  local_data/res/it/58.<tf>_regulon_pairwise_overlap.tsv    (the panel B statistics)
  local_data/fig/it/58.<tf>_regulon_target_overlap.pdf
"""

import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
# fonttype 42 embeds TrueType so the PDF keeps real text objects; the default (3) writes
# Type 3 glyph procedures, which Illustrator/Inkscape open as uneditable outlines
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')

INPUT_REGULON_TMPL = os.path.join(RES_DIR, '40.yoo25_{layer}_regulon_targets.tsv')
OUT_MEMBERSHIP_TMPL = os.path.join(RES_DIR, '58.{tf}_regulon_target_membership.tsv')
OUT_PAIRWISE_TMPL = os.path.join(RES_DIR, '58.{tf}_regulon_pairwise_overlap.tsv')
OUT_PDF_TMPL = os.path.join(FIG_DIR, '58.{tf}_regulon_target_overlap.pdf')

TF = 'Fosb'         # this script's own figure; 58b passes the other seven
SIGN = '+/+'        # activating regulons only, as in 41b.SIGN

# layer key (40's filename token) -> display label, in laminar-depth order
LAYER_LABEL = [('L2_3', 'L2/3'), ('L4', 'L4'), ('L5IT', 'L5IT'), ('L6IT', 'L6IT')]
LAYER_COLOR = {'L2/3': '#4c72b0', 'L4': '#dd8452', 'L5IT': '#55a868', 'L6IT': '#c44e52'}

STAR_FDR = 0.05
# Panel B's ramp. Symmetric about 0 because a pair CAN come in under its null -- two
# subclasses whose target sets avoid each other is as real a result as two that agree --
# and fixed rather than data-scaled so a colour means the same thing on every TF's page.
COLOR_ABS = 4.0
CORE_COLOR = '#333333'
GRAY = '#c8c8c8'

# UpSet geometry, in GridSpec width units: the bar columns get PATTERN_W each and any slack
# is parked in a spacer column, so a TF with 3 membership patterns draws them at the same
# density as one with 15 rather than stretching them across the page.
PATTERN_W, PATTERN_W_MIN, SETSIZE_W, TOP_W = 0.18, 0.9, 0.85, 3.55

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_sets(tf):
    """({label: targets}, {label: testable genes}) over the subclasses that have `tf`.

    The testable set is every gene called as a target of ANY activating regulon in that
    subclass's catalogue -- the background panel B's null draws from. It is a property of
    the catalogue, not of the TF, so it is read once here alongside the target sets.
    """
    targets, testable, absent = {}, {}, []
    for layer, label in LAYER_LABEL:
        path = INPUT_REGULON_TMPL.format(layer=layer)
        assert os.path.exists(path), f'missing {path}; run 40.yoo25_IT_regulon_targets.py first'
        tab = pd.read_csv(path, sep='\t')
        tab = tab[tab['regulation_direction'] == SIGN]
        sub = tab[tab['TF'] == tf]
        if sub.empty:
            absent.append(label)
            continue
        assert sub['regulon'].nunique() == 1, \
            f'{layer}: {tf} {SIGN} resolves to several regulons: {sorted(sub["regulon"].unique())}'
        testable[label] = set(tab['Gene'])
        targets[label] = set(sub['Gene'])
        print(f'  {label:<5} {len(targets[label]):>4} {tf} targets '
              f'({sub["source"].iloc[0]}), {len(testable[label])} testable genes')
    assert targets, f'the yoo25 catalogue has no {SIGN} {tf} regulon in ANY subclass'
    if absent:
        print(f'  no {SIGN} {tf} regulon called in: {", ".join(absent)}')
    return targets, testable, absent


def membership_table(targets):
    """One row per (gene, subclass) call, plus the gene's degree and pattern."""
    labels = list(targets)
    rows = []
    for gene in sorted(set().union(*targets.values())):
        pattern = [lbl for lbl in labels if gene in targets[lbl]]
        for lbl in pattern:
            rows.append(dict(gene=gene, subclass=lbl, degree=len(pattern),
                             pattern='|'.join(pattern)))
    return pd.DataFrame(rows)


def pairwise_stats(tf, targets, testable):
    """Observed overlap vs a hypergeometric null over the genes testable in both subclasses."""
    rows = []
    for a, b in itertools.combinations(list(targets), 2):
        shared = testable[a] & testable[b]
        # restricting to `shared` cannot drop a gene both subclasses called, since targets
        # of this regulon are by construction targets of some regulon
        ta, tb = targets[a] & shared, targets[b] & shared
        obs = len(ta & tb)
        exp = len(ta) * len(tb) / len(shared)
        rows.append(dict(
            TF=tf, subclass_a=a, subclass_b=b, n_a=len(targets[a]), n_b=len(targets[b]),
            n_testable_both=len(shared), n_a_testable=len(ta), n_b_testable=len(tb),
            overlap=obs, expected=exp,
            jaccard=len(targets[a] & targets[b]) / len(targets[a] | targets[b]),
            # +0.5 as in 54's log2_enr, so a zero overlap gives a finite (negative) value
            log2_enr=float(np.log2((obs + 0.5) / (exp + 0.5))),
            p=float(scipy.stats.hypergeom.sf(obs - 1, len(shared), len(ta), len(tb)))))
    out = pd.DataFrame(rows)
    if out.empty:      # a single-subclass regulon: no pair exists to test
        return out
    out['fdr'] = multipletests(out['p'], method='fdr_bh')[1]
    return out


def draw_upset(ax_bar, ax_dot, ax_set, table, targets):
    """Panel A: exclusive intersection sizes over every non-empty membership pattern."""
    labels = list(targets)
    counts = (table.drop_duplicates('gene')
                   .groupby(['pattern', 'degree']).size()
                   .reset_index(name='n'))
    # largest bar first; degree breaks ties so equal-sized patterns read simple-to-complex
    counts = counts.sort_values(['n', 'degree'], ascending=[False, True]).reset_index(drop=True)
    x = np.arange(len(counts))

    bars = ax_bar.bar(x, counts['n'], width=0.62, color='0.25', zorder=3)
    for rect, n in zip(bars, counts['n']):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, n, str(n),
                    ha='center', va='bottom', fontsize=7)
    ax_bar.set_ylabel('genes in exactly\nthis combination', fontsize=8)
    ax_bar.set_xlim(-0.7, len(counts) - 0.3)
    ax_bar.set_ylim(0, counts['n'].max() * 1.18)
    ax_bar.set_xticks([])
    ax_bar.tick_params(labelsize=7, length=2)
    ax_bar.grid(True, axis='y', color='0.90', lw=0.5, zorder=0)
    ax_bar.set_axisbelow(True)
    for side in ('top', 'right', 'bottom'):
        ax_bar.spines[side].set_visible(False)

    for i, (_j, row) in enumerate(counts.iterrows()):
        members = row['pattern'].split('|')
        ys = [labels.index(m) for m in members]
        ax_dot.scatter([i] * len(labels), range(len(labels)), s=42, color=GRAY, zorder=2)
        if len(ys) > 1:
            ax_dot.plot([i, i], [min(ys), max(ys)], color='0.25', lw=1.4, zorder=3)
        ax_dot.scatter([i] * len(ys), ys, s=42,
                       color=[LAYER_COLOR[m] for m in members], zorder=4)
    ax_dot.set_xlim(-0.7, len(counts) - 0.3)
    ax_dot.set_ylim(len(labels) - 0.4, -0.6)
    ax_dot.set_xticks([])
    ax_dot.set_yticks(range(len(labels)))
    ax_dot.set_yticklabels(labels, fontsize=8)
    ax_dot.tick_params(length=0)
    for side in ax_dot.spines:
        ax_dot.spines[side].set_visible(False)

    sizes = [len(targets[lbl]) for lbl in labels]
    ax_set.barh(range(len(labels)), sizes, height=0.55,
                color=[LAYER_COLOR[lbl] for lbl in labels], zorder=3)
    for i, n in enumerate(sizes):
        ax_set.text(n + 0.02 * max(sizes), i, str(n), va='center', ha='left', fontsize=7)
    ax_set.set_ylim(len(labels) - 0.4, -0.6)
    ax_set.set_xlim(max(sizes) * 1.30, 0)          # bars grow leftwards, toward the matrix
    ax_set.set_yticks([])
    ax_set.set_xlabel('targets in the subclass', fontsize=8, labelpad=2)
    ax_set.tick_params(labelsize=7, length=2)
    for side in ('top', 'left', 'right'):
        ax_set.spines[side].set_visible(False)


def draw_pairwise(ax, cax, pw, targets):
    """Panel B: the overlap matrix, coloured by enrichment over the shared-testable null."""
    labels = list(targets)
    n = len(labels)
    grid = np.full((n, n), np.nan)
    for _i, r in pw.iterrows():
        a, b = labels.index(r['subclass_a']), labels.index(r['subclass_b'])
        grid[a, b] = grid[b, a] = r['log2_enr']

    norm = Normalize(vmin=-COLOR_ABS, vmax=COLOR_ABS)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap='RdBu_r', norm=norm)
    im.cmap.set_bad('white')

    for _i, r in pw.iterrows():
        a, b = labels.index(r['subclass_a']), labels.index(r['subclass_b'])
        star = '*' if r['fdr'] < STAR_FDR else ''
        # black on the pale middle of RdBu_r, white where the ramp saturates
        color = 'white' if abs(r['log2_enr']) > 0.62 * COLOR_ABS else 'black'
        for i, j in [(a, b), (b, a)]:
            ax.text(j, i, f'{r["overlap"]:.0f}{star}\nJ={r["jaccard"]:.2f}',
                    ha='center', va='center', fontsize=7, color=color, linespacing=1.25)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        tick.set_color(LAYER_COLOR[lbl])
    for tick, lbl in zip(ax.get_yticklabels(), labels):
        tick.set_color(LAYER_COLOR[lbl])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=2)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title('shared targets (* FDR < %g)\nJ = Jaccard index' % STAR_FDR,
                 fontsize=7.5, pad=5)

    cb = plt.colorbar(im, cax=cax)
    cb.set_label('log2 obs/exp over genes\ntestable in both', fontsize=7)
    cb.ax.tick_params(labelsize=7)


def draw_core(ax, tf, table, targets, absent):
    """Panel C: the genes every subclass called, and how the union splits by degree."""
    labels = list(targets)
    genes = sorted(table.loc[table['degree'] == len(labels), 'gene'].unique())
    union = table['gene'].nunique()
    by_deg = table.drop_duplicates('gene')['degree'].value_counts().sort_index()

    ax.axis('off')
    lines = [f'union of the {len(labels)} {tf} target sets: {union} genes']
    if absent:
        lines += [f'(no {tf} regulon in {", ".join(absent)})']
    lines += ['']
    if len(labels) == 1:
        # one catalogue: every gene is trivially "shared by all of them" and there is no
        # pair to test, so say that instead of printing a 100%-degree-1 breakdown
        lines += [f'only {labels[0]} has this regulon, so there is no overlap',
                  'between subclasses to measure -- its targets are listed',
                  'in the membership table.']
        ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=8, family='monospace',
                transform=ax.transAxes)
        return
    lines += [f'  in {d} of {len(labels)} subclasses: {by_deg.get(d, 0):>4}  '
              f'({100 * by_deg.get(d, 0) / union:>3.0f}%)' for d in range(1, len(labels) + 1)]
    lines += ['', f'called in all {len(labels)} ({len(genes)}):']
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=8, family='monospace',
            transform=ax.transAxes)

    # the core can run to hundreds of genes for a broad regulon (Egr1); print what fits and
    # say how many were cut rather than silently truncating -- the full list is in the TSV
    per_row, max_rows = 5, 12
    rows = [' '.join(f'{g:<11}' for g in genes[i:i + per_row])
            for i in range(0, len(genes), per_row)]
    shown = rows[:max_rows]
    if len(rows) > max_rows:
        shown.append(f'... and {len(genes) - max_rows * per_row} more '
                     f'(see the membership table)')
    ax.text(0, 1 - 0.052 * (len(lines) + 0.6), '\n'.join(shown), va='top', ha='left',
            fontsize=7.5, family='monospace', color=CORE_COLOR, transform=ax.transAxes)


def build_figure(tf, targets, absent, table, pw):
    """The three-panel page for one TF."""
    n_patterns = table['pattern'].nunique()
    bars_w = max(PATTERN_W * n_patterns, PATTERN_W_MIN)
    slack = max(TOP_W - SETSIZE_W - bars_w, 1e-3)

    # nested grids rather than one 3-row GridSpec: the bar and dot rows of the UpSet must
    # touch (they share an x axis), while the bottom row needs room for its own tick labels
    fig = plt.figure(figsize=(9.2, 8.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.34)
    gs_top = gs[0].subgridspec(2, 3, height_ratios=[1.7, 0.85],
                               width_ratios=[SETSIZE_W, bars_w, slack],
                               hspace=0.06, wspace=0.05)
    ax_bar = fig.add_subplot(gs_top[0, 1])
    ax_dot = fig.add_subplot(gs_top[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs_top[1, 0], sharey=ax_dot)
    draw_upset(ax_bar, ax_dot, ax_set, table, targets)

    gs_bot = gs[1].subgridspec(1, 3, width_ratios=[1.15, 0.06, 2.0], wspace=0.34)
    ax_core = fig.add_subplot(gs_bot[0, 2])
    if pw.empty:
        # one subclass, so panels A and B degenerate; C still carries the counts
        fig.add_subplot(gs_bot[0, 0]).axis('off')
        fig.add_subplot(gs_bot[0, 1]).axis('off')
    else:
        draw_pairwise(fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1]), pw, targets)
    draw_core(ax_core, tf, table, targets, absent)

    heads = [(0.02, 0.955, f'A  {tf} regulon targets across IT subclasses'),
             (0.42, 0.455, 'C  the shared core')]
    if not pw.empty:
        heads.append((0.02, 0.455, 'B  pairwise overlap'))
    for x, y, s in heads:
        fig.text(x, y, s, fontsize=10, fontweight='bold')
    fig.suptitle(f'One TF, four catalogues: SCENIC+ {tf} ({SIGN}) targets are inferred '
                 f'independently per subclass (yoo25)', fontsize=9, y=1.0)
    return fig


def main(tf=TF, save=True):
    """Run one TF; returns (membership table, pairwise stats, figure) for 58b to reuse."""
    print(f'{tf} {SIGN} regulon target sets, yoo25 per-subclass catalogues:')
    targets, testable, absent = load_sets(tf)
    table = membership_table(targets)
    pw = pairwise_stats(tf, targets, testable)

    union = table['gene'].nunique()
    core = int((table.drop_duplicates('gene')['degree'] == len(targets)).sum())
    private = int((table.drop_duplicates('gene')['degree'] == 1).sum())
    print(f'  union {union} genes; {core} called in all {len(targets)}, {private} in exactly '
          f'one ({100 * private / union:.0f}%)')
    for _i, r in pw.iterrows():
        print(f'  {r["subclass_a"]:<5} vs {r["subclass_b"]:<5} overlap {r["overlap"]:>4} '
              f'(exp {r["expected"]:6.2f}, log2 {r["log2_enr"]:+5.2f}, FDR {r["fdr"]:.2e}, '
              f'J={r["jaccard"]:.3f})')

    fig = build_figure(tf, targets, absent, table, pw)
    if save:
        out_mem = OUT_MEMBERSHIP_TMPL.format(tf=tf.lower())
        out_pw = OUT_PAIRWISE_TMPL.format(tf=tf.lower())
        out_pdf = OUT_PDF_TMPL.format(tf=tf.lower())
        table.to_csv(out_mem, sep='\t', index=False)
        pw.to_csv(out_pw, sep='\t', index=False)
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)
        for path in (out_pdf, out_mem, out_pw):
            print(f'  wrote {path}')
    return table, pw, fig


if __name__ == '__main__':
    main()
