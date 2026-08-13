"""Per-gene temporal shape behind each archetype line of scripts/it_evo/25.

Reads the table written by scripts/it_evo/25c (run it first) and draws, per panel, one
heatmap per archetype: rows are the archetype's genes ordered by 25c's rank (most dynamic on
top), columns are the four developmental stages, color is that gene's per-stage score divided
by its own peak stage — so every row spans 0..1 and rows are comparable in shape regardless
of expression level. A light cell is a stage where the gene is off; a dark cell is its peak.

Beside each heatmap is the archetype's own curve — the mean over its genes of the per-stage
scores, divided by its largest per-stage mean. That is exactly the line script 25 plots, so
the heatmap and the curve are the same numbers at two levels of aggregation: the curve is a
column mean of the heatmap's *unscaled* values.

Gene labels carry 25c's range_over_sem in parentheses, the ranking metric.

Reads:
  local_data/res/it_evo/25c.wang25_multiome_it_archetype_gene_dynamics.tsv
Outputs:
  local_data/fig/it_evo/25d.{L23,L4,L5IT,L6IT}_archetype_gene_dynamics.pdf
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42     # editable vector text in PDF
plt.rcParams['svg.fonttype'] = 'none'  # editable vector text in SVG

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_TSV      = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo',
                           '25c.wang25_multiome_it_archetype_gene_dynamics.tsv')
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
OUT_PDF_TMPL = os.path.join(OUT_FIG_DIR, '25d.{token}_archetype_gene_dynamics.pdf')

# Panel label (as written by 25c) -> filename token, in the order of script 25's panels.
PANEL_TOKENS = {'L2/3': 'L23', 'L4': 'L4', 'L5 IT': 'L5IT', 'L6 IT': 'L6IT'}

STAGE_ORDER  = ['Second_trimester', 'Third_trimester', 'Infancy', 'Adolescence']
STAGE_LABELS = {'Second_trimester': 'Second\ntrimester',
                'Third_trimester': 'Third\ntrimester',
                'Infancy': 'Infancy',
                'Adolescence': 'Adolescence'}

# Sequential magnitude (off -> peak): one hue, light -> dark. Not a rainbow.
CMAP        = 'Blues'
CURVE_COLOR = '#08519c'    # the dark end of the same ramp
ROW_HEIGHT  = 0.13         # inches per gene row
GENE_FONT   = 5.5
HEAT_W      = 2.3          # inches, heatmap column
CURVE_W     = 1.5          # inches, curve column


def panel_figure(panel, token, df):
    """One figure for a panel: an archetype per row, heatmap + its script-25 curve."""
    letters = sorted(df['archetype'].unique())
    blocks = {L: df[df['archetype'] == L].sort_values('rank') for L in letters}
    n_rows = [len(blocks[L]) for L in letters]

    fig_h = sum(n_rows) * ROW_HEIGHT + 1.5 * len(letters) + 0.9
    fig = plt.figure(figsize=(HEAT_W + CURVE_W + 1.7, fig_h))
    gs = fig.add_gridspec(len(letters), 2, height_ratios=n_rows, width_ratios=[HEAT_W, CURVE_W],
                          hspace=0.35, wspace=0.55, left=0.34, right=0.97,
                          top=1 - 0.5 / fig_h, bottom=0.9 / fig_h)

    norm_cols = [f'norm_{s}' for s in STAGE_ORDER]
    xpos = np.arange(len(STAGE_ORDER))
    im = None

    for row, L in enumerate(letters):
        blk = blocks[L]
        vals = blk[norm_cols].to_numpy(dtype=float)          # genes x stages
        peak = vals.max(axis=1, keepdims=True)
        if np.any(peak <= 0):
            raise ValueError(f"{panel} {L}: gene with a non-positive peak; cannot scale rows")
        shape = vals / peak                                   # each row 0..1, peak = 1

        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(shape, aspect='auto', cmap=CMAP, vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_yticks(np.arange(len(blk)))
        ax.set_yticklabels([f'{g} ({v:.0f})' for g, v in
                            zip(blk['human_gene'], blk['range_over_sem'])], fontsize=GENE_FONT)
        ax.set_xticks(xpos)
        ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=6,
                           rotation=45, ha='right')
        ax.tick_params(length=2, pad=1.5)
        ax.set_title(f'{panel} — archetype {L}  ({len(blk)} genes)', fontsize=8, pad=4)
        for side in ax.spines.values():
            side.set_visible(False)

        # --- the archetype's own line from script 25: gene-mean, then divide by its max ---
        mean_by_stage = vals.mean(axis=0)
        curve = mean_by_stage / mean_by_stage.max()
        axc = fig.add_subplot(gs[row, 1])
        axc.plot(xpos, curve, '-o', color=CURVE_COLOR, lw=1.4, markersize=3.5)
        # The heatmap's height scales with the gene count; keep the curve square so its slope
        # is not exaggerated for the archetypes with the most genes.
        axc.set_box_aspect(1)
        axc.set_ylim(0, 1.12)
        axc.set_xlim(-0.4, len(STAGE_ORDER) - 0.6)
        axc.set_xticks(xpos)
        axc.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], fontsize=6,
                            rotation=45, ha='right')
        axc.tick_params(labelsize=6, length=2, pad=1.5)
        axc.set_ylabel('archetype score\n(script 25 line)', fontsize=6)
        axc.spines['top'].set_visible(False)
        axc.spines['right'].set_visible(False)

    cax = fig.add_axes([0.34, 0.30 / fig_h, 0.40, 0.10 / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('per-gene score / that gene\'s peak stage', fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)

    out_pdf = OUT_PDF_TMPL.format(token=token)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_pdf


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    print(f'Loading {IN_TSV}')
    df = pd.read_csv(IN_TSV, sep='\t')
    missing = set(PANEL_TOKENS) - set(df['panel'])
    if missing:
        raise ValueError(f"Panels missing from {IN_TSV}: {missing} — rerun script 25c")

    for panel, token in PANEL_TOKENS.items():
        sub = df[df['panel'] == panel]
        out_pdf = panel_figure(panel, token, sub)
        print(f'[{panel}] {len(sub)} genes across {sub["archetype"].nunique()} archetypes '
              f'-> {out_pdf}')


if __name__ == '__main__':
    main()
