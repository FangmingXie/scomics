"""yoo25 NR embedding grid coloured by ONE archetype-score channel per row.

61.v2 colours each cell by script 59's simplex blend: all NOC scores are rescaled, normalised to
sum to 1, mixed into a hue, then desaturated by the winning margin. That is faithful but hard to
decode -- a given grey could mean "high on everything" or "low on everything", and the hue mixes
three channels at once.

This figure splits the colour into two channels and gives each its OWN ROW, so nothing is mixed
and each panel is a single-quantity readout:

    row 1 of each subclass   d = score_C - score_A   diverging, which flanking archetype wins
    row 2 of each subclass   b = score_B             sequential, how much of the middle archetype

`d` ramps between the A and C archetype colours through neutral grey; `b` ramps from neutral
grey to the B colour. No simplex normalisation, no margin desaturation, no blending of the two
channels against each other -- a cell's colour reads off exactly one number, and each row
carries its own colour bar.

Colours follow the DISPLAYED primed labels (ARCHETYPE_MAPPING.md), so the diverging ends flip
between subclasses: L2/3 and L4 are letter reversals (internal A = C', internal C = A') while
L6IT is identity. That is handled by reading the depth arc, never by hard-coding a direction.

L5IT is NOC=2 and has no score_C, so its diverging row is `d = score_B - score_A`. Its second
row still shows `score_B`, but note that means something different there: with only two
archetypes, score_B is a flanking archetype (A'), not a middle one. Both row labels say so.

Everything else matches 61.v2's grid: ages left to right, only that panel's cells drawn, axis
limits shared across a subclass's row pair, and the P21 archetype simplex (script 62) in black on
the P21 panel alone. No recomputation -- the cached 61.v2 / 62 TSVs are read as-is.

Like 61.v2's panels, each channel is rescaled WITHIN each age, so colours are not comparable
between age columns; the cross-age quantity is 61.v2's pooled weight trajectory.

Reads:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
  local_data/res/it/61.v2.yoo25_nr_<token>_archetype_scores.tsv
  local_data/res/it/62.yoo25_p21_<token>_archetype_coords.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/fig/it/61.v3.yoo25_nr_score_channel_embedding.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_ARCMAP  = os.path.join(IT_EVO_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
IN_COORDS  = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_pc_coords.tsv')
IN_SCORES  = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_archetype_scores.tsv')
IN_ARCH    = os.path.join(RES_DIR, '62.yoo25_p21_{token}_archetype_coords.tsv')
OUT_PDF    = os.path.join(FIG_DIR, '61.v3.yoo25_nr_score_channel_embedding.pdf')

SUBCLASSES = [
    dict(subclass='L2/3', token='L23'),
    dict(subclass='L4',   token='L4'),
    dict(subclass='L5IT', token='L5IT'),
    dict(subclass='L6IT', token='L6IT'),
]

# --- parameters ---
AGE_COL      = 'Age'
AGES         = ['P6', 'P8', 'P10', 'P12', 'P14', 'P17', 'P21']
REF_AGE      = 'P21'
ARCH_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
# Colour follows the DISPLAYED (primed) label, never the internal key -- ARCHETYPE_MAPPING.md.
ARCH_COLORS  = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
NEUTRAL_RGB  = np.array(mcolors.to_rgb('#e6e6e6'))   # d == 0 and b == 0
DIV_PCTILE   = 95     # |d| percentile that saturates the diverging ramp
SEQ_PCTILE   = (5, 95)
POINT_SIZE   = 2.5
KEY_N        = 128    # colour-bar raster resolution
FIG_PANEL_W  = 2.4
FIG_PANEL_H  = 2.4
DPI          = 300


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


def diverging_rgb(d_scaled, rgb_neg, rgb_pos):
    """d in [-1,1] -> RGB ramping rgb_neg <- neutral -> rgb_pos."""
    end = np.where(d_scaled[:, None] >= 0, rgb_pos[None, :], rgb_neg[None, :])
    return NEUTRAL_RGB + (end - NEUTRAL_RGB) * np.abs(d_scaled)[:, None]


def sequential_rgb(b_scaled, rgb_b):
    """b in [0,1] -> RGB ramping neutral -> rgb_b."""
    return NEUTRAL_RGB + (rgb_b[None, :] - NEUTRAL_RGB) * b_scaled[:, None]


def scale_diverging(d):
    """Symmetric rescale to [-1,1] on this cell set's own spread."""
    lim = np.percentile(np.abs(d), DIV_PCTILE)
    if lim <= 0:
        raise ValueError(f'degenerate diverging channel: |d| {DIV_PCTILE}th percentile is {lim}')
    return np.clip(d / lim, -1.0, 1.0)


def scale_sequential(b):
    """Rescale to [0,1] on this cell set's own spread."""
    lo, hi = np.percentile(b, SEQ_PCTILE)
    if hi <= lo:
        raise ValueError(f'degenerate sequential channel: percentiles {SEQ_PCTILE} are {lo}, {hi}')
    return np.clip((b - lo) / (hi - lo), 0.0, 1.0)


def draw_key(ax, channel):
    """A horizontal colour bar for one channel, drawn as a band inside an empty panel."""
    if channel['kind'] == 'diverging':
        grid = np.linspace(-1, 1, KEY_N)
        rgb = diverging_rgb(grid, channel['rgb_neg'], channel['rgb_pos'])
        ticks, labels = [-1, 0, 1], [channel['neg_label'], '0', channel['pos_label']]
        lo_hi = (-1, 1)
    else:
        grid = np.linspace(0, 1, KEY_N)
        rgb = sequential_rgb(grid, channel['rgb_b'])
        ticks, labels = [0, 1], ['low', 'high']
        lo_hi = (0, 1)
    ax.imshow(rgb[None, :, :], origin='lower', aspect='auto',
              extent=[lo_hi[0], lo_hi[1], 0.42, 0.58])
    ax.set_xlim(*lo_hi); ax.set_ylim(0, 1)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel(channel['expr'], fontsize=8)
    ax.set_title('colour key', fontsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


print('--- yoo25 NR embedding, one score channel per row ---')
os.makedirs(FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:                        # fail fast, before any plotting
    for tmpl in (IN_COORDS, IN_SCORES, IN_ARCH):
        path = tmpl.format(token=cfg['token'])
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path} -- run 61.v2 then 62 first')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')

relabel_by_token = load_relabel()
rows = []                                     # one entry per (subclass, channel) = one figure row

for cfg in SUBCLASSES:
    S, token = cfg['subclass'], cfg['token']
    coords = pd.read_csv(IN_COORDS.format(token=token), sep='\t', index_col=0)
    scores = pd.read_csv(IN_SCORES.format(token=token), sep='\t', index_col=0).reindex(coords.index)
    if scores.isna().any().any():
        raise ValueError(f'{S}: cells missing from {IN_SCORES.format(token=token)}')

    letters = [L for L in ARCH_LETTERS if f'score_{L}' in scores.columns]
    relabel = relabel_by_token[token]
    if set(relabel) != set(letters):
        raise ValueError(f'{S}: arc-table letters {sorted(relabel)} != score letters {letters}')

    # NOC=3 -> the diverging axis runs between the two flanking archetypes, A and C. NOC=2 has
    # no middle archetype, so it runs A -> B instead and score_B is a FLANK, not a middle.
    if len(letters) == 3:
        pos_letter, b_note = 'C', 'middle archetype'
    elif len(letters) == 2:
        pos_letter, b_note = 'B', 'flanking archetype (NOC=2, no middle)'
    else:
        raise ValueError(f'{S}: NOC={len(letters)} is not supported by this figure')

    ages = coords[AGE_COL].astype(str).values
    missing_ages = sorted(set(AGES) - set(ages))
    if missing_ages:
        raise ValueError(f'{S}: no cells at age(s) {missing_ages}')

    arch = pd.read_csv(IN_ARCH.format(token=token), sep='\t', index_col=0)
    if list(arch.index) != letters:
        raise ValueError(f'{S}: {IN_ARCH.format(token=token)} index {list(arch.index)} '
                         f'!= score letters {letters}')

    rgb_neg = np.array(mcolors.to_rgb(ARCH_COLORS[relabel['A']]))
    rgb_pos = np.array(mcolors.to_rgb(ARCH_COLORS[relabel[pos_letter]]))
    rgb_b   = np.array(mcolors.to_rgb(ARCH_COLORS[relabel['B']]))

    # Rescale WITHIN each age, matching 61.v2's per-age panel colour.
    d_raw = scores[f'score_{pos_letter}'].values - scores['score_A'].values
    b_raw = scores['score_B'].values
    rgb_d = np.zeros((len(coords), 3))
    rgb_b_cells = np.zeros((len(coords), 3))
    for age in AGES:
        m = ages == age
        rgb_d[m] = diverging_rgb(scale_diverging(d_raw[m]), rgb_neg, rgb_pos)
        rgb_b_cells[m] = sequential_rgb(scale_sequential(b_raw[m]), rgb_b)

    shared = dict(subclass=S, letters=letters, relabel=relabel,
                  pcs=coords[['PC1', 'PC2']].values, ages=ages,
                  vertices=arch[['PC1', 'PC2']].values,
                  n_by_age={age: int((ages == age).sum()) for age in AGES})
    rows.append(dict(shared, kind='diverging', rgb=rgb_d, first_of_pair=True,
                     rgb_neg=rgb_neg, rgb_pos=rgb_pos,
                     neg_label=relabel['A'], pos_label=relabel[pos_letter],
                     expr=f'score_{pos_letter} - score_A',
                     row_label=f'{S}\nscore_{pos_letter} - score_A\n'
                               f'({relabel["A"]} ↔ {relabel[pos_letter]})'))
    rows.append(dict(shared, kind='sequential', rgb=rgb_b_cells, first_of_pair=False,
                     rgb_b=rgb_b, expr='score_B',
                     row_label=f'{S}\nscore_B ({relabel["B"]})'))
    print(f'  {S:5s} NOC={len(letters)}  diverging: score_{pos_letter} - score_A '
          f'({relabel["A"]} <-> {relabel[pos_letter]})   sequential: score_B '
          f'({relabel["B"]}, {b_note})')

# ===================== figure: (subclass x channel) rows x age cols, plus a key column ========

plt.rcParams['pdf.fonttype'] = 42             # editable vector text
ncol = len(AGES) + 1                          # last column is the per-row colour key
fig, axes = plt.subplots(len(rows), ncol, squeeze=False,
                         figsize=(FIG_PANEL_W * ncol, FIG_PANEL_H * len(rows)))

for row, R in enumerate(rows):
    xy = R['pcs']
    extent = np.vstack([xy, R['vertices']])
    pad = 0.04 * (extent.max(axis=0) - extent.min(axis=0))
    xlim = (extent[:, 0].min() - pad[0], extent[:, 0].max() + pad[0])
    ylim = (extent[:, 1].min() - pad[1], extent[:, 1].max() + pad[1])

    for col, age in enumerate(AGES):
        ax = axes[row][col]
        this = R['ages'] == age
        ax.scatter(xy[this, 0], xy[this, 1], s=POINT_SIZE, c=R['rgb'][this],
                   linewidths=0, zorder=2, rasterized=True)

        # Simplex on the P21 panel only -- the one age it was fit on -- and all in black.
        if age == REF_AGE:
            vx = R['vertices']
            closed = np.vstack([vx, vx[0]]) if len(vx) > 2 else vx
            ax.plot(closed[:, 0], closed[:, 1], '-', color='black', linewidth=0.9, zorder=3)
            ax.scatter(vx[:, 0], vx[:, 1], marker='D', c='black', s=36, zorder=4)
            for (ax_, ay_), L in zip(vx, R['letters']):
                ax.annotate(R['relabel'][L], (ax_, ay_), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, fontweight='bold', color='black', zorder=5)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if row == len(rows) - 1:
            ax.set_xlabel('PC1')
        if col == 0:
            ax.set_ylabel(R['row_label'], fontweight='bold', fontsize=8)
        # Age titles head each subclass's row PAIR; the second row inherits the column.
        if R['first_of_pair']:
            basis = ' (basis)' if age == REF_AGE else ''
            ax.set_title(f'{age}  n={R["n_by_age"][age]}{basis}', fontsize=9)
        sns.despine(ax=ax)

    draw_key(axes[row][ncol - 1], R)

fig.suptitle('yoo25 IT NR series in the P21 PC basis — one archetype-score channel per row '
             '(diverging score_C − score_A, then score_B); rescaled within each age',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'\nSaved {OUT_PDF}')
print('\nDone.')
