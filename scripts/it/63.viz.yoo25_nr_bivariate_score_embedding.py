"""yoo25 NR embedding grid recoloured by TWO archetype-score channels instead of the 3-way blend.

61.v2 colours each cell by script 59's simplex blend: all NOC scores are rescaled, normalised to
sum to 1, mixed into a hue, then desaturated by the winning margin. That is faithful but hard to
decode -- a given grey could mean "high on everything" or "low on everything", and the hue mixes
three channels at once.

This figure simplifies the rule to two interpretable numbers per cell:

    x-channel  d = score_C - score_A     signed, which of the two flanking archetypes wins
    y-channel  b = score_B               unsigned, how much of the middle archetype

Colour is then a bivariate map: `d` drives a diverging ramp between the A and C archetype
colours through a neutral grey, and `b` mixes that result toward the B colour. No normalisation
to a simplex, no margin-based desaturation -- a cell's colour is a direct readout of two
quantities, and each row carries a 2-D colour key showing exactly that square.

Colours follow the DISPLAYED primed labels (ARCHETYPE_MAPPING.md), so the diverging ends flip
between subclasses: L2/3 and L4 are letter reversals (internal A = C', internal C = A') while
L6IT is identity. That is handled by reading the depth arc, never by hard-coding a direction.

L5IT is NOC=2 and has no score_C, so `d = score_B - score_A` there and the b-channel is held at
neutral -- its row is a pure diverging ramp. Its x axis therefore means something different from
the other three rows, which its colour key states explicitly.

Everything else matches 61.v2's grid: subclass per row, ages left to right, only that panel's
cells drawn, axis limits shared across the row, and the P21 archetype simplex (script 62) drawn
in black on the P21 panel alone. No recomputation -- the cached 61.v2 / 62 TSVs are read as-is.

Like 61.v2's panels, the colour rescale runs WITHIN each age, so colours are not comparable
between age columns; the cross-age quantity is 61.v2's pooled weight trajectory.

Reads:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
  local_data/res/it/61.v2.yoo25_nr_<token>_archetype_scores.tsv
  local_data/res/it/62.yoo25_p21_<token>_archetype_coords.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/fig/it/63.yoo25_nr_bivariate_score_embedding.pdf
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
OUT_PDF    = os.path.join(FIG_DIR, '63.yoo25_nr_bivariate_score_embedding.pdf')

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
B_PCTILE     = (5, 95)
B_MAX        = 0.70   # cap on the mix toward the B colour, so `d` is never fully overwritten
POINT_SIZE   = 2.5
KEY_N        = 64     # colour-key raster resolution
FIG_PANEL_W  = 2.7
FIG_PANEL_H  = 2.7
DPI          = 300


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


def bivariate_rgb(d_scaled, b_scaled, rgb_neg, rgb_pos, rgb_b):
    """(d in [-1,1], b in [0,1]) -> RGB.

    `d` ramps from the negative archetype's colour through neutral grey to the positive one;
    `b` then mixes that result toward the B colour. Both inputs are already rescaled.
    """
    end = np.where(d_scaled[:, None] >= 0, rgb_pos[None, :], rgb_neg[None, :])
    div = NEUTRAL_RGB + (end - NEUTRAL_RGB) * np.abs(d_scaled)[:, None]
    return div + (rgb_b[None, :] - div) * b_scaled[:, None]


def scale_channels(scores, letters, pos_letter, b_letter):
    """Per-cell (d, b) rescaled to [-1,1] and [0,B_MAX] using THIS cell set's own spread."""
    d = scores[f'score_{pos_letter}'].values - scores['score_A'].values
    lim = np.percentile(np.abs(d), DIV_PCTILE)
    if lim <= 0:
        raise ValueError(f'degenerate d channel: |d| {DIV_PCTILE}th percentile is {lim}')
    d_scaled = np.clip(d / lim, -1.0, 1.0)

    if b_letter is None:                      # NOC=2: no middle archetype, pure diverging ramp
        return d_scaled, np.zeros_like(d_scaled)
    b = scores[f'score_{b_letter}'].values
    lo, hi = np.percentile(b, B_PCTILE)
    if hi <= lo:
        raise ValueError(f'degenerate b channel: percentiles {B_PCTILE} are {lo}, {hi}')
    return d_scaled, np.clip((b - lo) / (hi - lo), 0.0, 1.0) * B_MAX


def draw_key(ax, S):
    """The 2-D colour square this row's panels are read against."""
    dd, bb = np.meshgrid(np.linspace(-1, 1, KEY_N), np.linspace(0, B_MAX, KEY_N))
    rgb = bivariate_rgb(dd.ravel(), bb.ravel(), S['rgb_neg'], S['rgb_pos'], S['rgb_b'])
    ax.imshow(rgb.reshape(KEY_N, KEY_N, 3), origin='lower', aspect='auto',
              extent=[-1, 1, 0, B_MAX])
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([S['relabel']['A'], '0', S['relabel'][S['pos_letter']]], fontsize=7)
    ax.set_xlabel(f'score_{S["pos_letter"]} - score_A', fontsize=7)
    if S['b_letter'] is None:
        ax.set_yticks([])
        ax.set_ylabel('(NOC=2: no B)', fontsize=6.5)
    else:
        ax.set_yticks([0, B_MAX])
        ax.set_yticklabels(['low', 'high'], fontsize=7)
        ax.set_ylabel(f'score_{S["b_letter"]}  ({S["relabel"][S["b_letter"]]})', fontsize=7)
    ax.set_title('colour key', fontsize=8)


print('--- yoo25 NR embedding recoloured by two score channels ---')
os.makedirs(FIG_DIR, exist_ok=True)

for cfg in SUBCLASSES:                        # fail fast, before any plotting
    for tmpl in (IN_COORDS, IN_SCORES, IN_ARCH):
        path = tmpl.format(token=cfg['token'])
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path} -- run 61.v2 then 62 first')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')

relabel_by_token = load_relabel()
panels = []

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

    # NOC=3 -> d = C - A with B as the second channel. NOC=2 has no middle archetype, so the
    # diverging axis runs A -> B and the second channel is held flat.
    if len(letters) == 3:
        pos_letter, b_letter = 'C', 'B'
    elif len(letters) == 2:
        pos_letter, b_letter = 'B', None
    else:
        raise ValueError(f'{S}: NOC={len(letters)} is not supported by this two-channel figure')

    ages = coords[AGE_COL].astype(str).values
    missing_ages = sorted(set(AGES) - set(ages))
    if missing_ages:
        raise ValueError(f'{S}: no cells at age(s) {missing_ages}')

    # Rescale WITHIN each age, matching 61.v2's per-age panel colour.
    rgb_neg = np.array(mcolors.to_rgb(ARCH_COLORS[relabel['A']]))
    rgb_pos = np.array(mcolors.to_rgb(ARCH_COLORS[relabel[pos_letter]]))
    rgb_b   = np.array(mcolors.to_rgb(ARCH_COLORS[relabel[b_letter]])) if b_letter else NEUTRAL_RGB
    rgb = np.zeros((len(coords), 3))
    for age in AGES:
        m = ages == age
        d_s, b_s = scale_channels(scores[m], letters, pos_letter, b_letter)
        rgb[m] = bivariate_rgb(d_s, b_s, rgb_neg, rgb_pos, rgb_b)

    arch = pd.read_csv(IN_ARCH.format(token=token), sep='\t', index_col=0)
    if list(arch.index) != letters:
        raise ValueError(f'{S}: {IN_ARCH.format(token=token)} index {list(arch.index)} '
                         f'!= score letters {letters}')

    panels.append(dict(subclass=S, token=token, letters=letters, relabel=relabel,
                       pos_letter=pos_letter, b_letter=b_letter,
                       rgb_neg=rgb_neg, rgb_pos=rgb_pos, rgb_b=rgb_b,
                       pcs=coords[['PC1', 'PC2']].values, ages=ages, rgb=rgb,
                       vertices=arch[['PC1', 'PC2']].values,
                       n_by_age={age: int((ages == age).sum()) for age in AGES}))
    print(f'  {S:5s} NOC={len(letters)}  d = score_{pos_letter} - score_A '
          f'({relabel["A"]} <-> {relabel[pos_letter]})'
          + (f'  b = score_{b_letter} ({relabel[b_letter]})' if b_letter else '  b = n/a'))

# ===================== figure: subclass (row) x age (col), plus a colour key column ===========

plt.rcParams['pdf.fonttype'] = 42             # editable vector text
ncol = len(AGES) + 1                          # last column is the per-row colour key
fig, axes = plt.subplots(len(panels), ncol, squeeze=False,
                         figsize=(FIG_PANEL_W * ncol, FIG_PANEL_H * len(panels)))

for row, P in enumerate(panels):
    xy = P['pcs']
    extent = np.vstack([xy, P['vertices']])
    pad = 0.04 * (extent.max(axis=0) - extent.min(axis=0))
    xlim = (extent[:, 0].min() - pad[0], extent[:, 0].max() + pad[0])
    ylim = (extent[:, 1].min() - pad[1], extent[:, 1].max() + pad[1])

    for col, age in enumerate(AGES):
        ax = axes[row][col]
        this = P['ages'] == age
        ax.scatter(xy[this, 0], xy[this, 1], s=POINT_SIZE, c=P['rgb'][this],
                   linewidths=0, zorder=2, rasterized=True)

        # Simplex on the P21 panel only -- the one age it was fit on -- and all in black.
        if age == REF_AGE:
            vx = P['vertices']
            closed = np.vstack([vx, vx[0]]) if len(vx) > 2 else vx
            ax.plot(closed[:, 0], closed[:, 1], '-', color='black', linewidth=0.9, zorder=3)
            ax.scatter(vx[:, 0], vx[:, 1], marker='D', c='black', s=42, zorder=4)
            for (ax_, ay_), L in zip(vx, P['letters']):
                ax.annotate(P['relabel'][L], (ax_, ay_), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, fontweight='bold', color='black', zorder=5)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if row == len(panels) - 1:
            ax.set_xlabel('PC1')
        if col == 0:
            ax.set_ylabel(f'{P["subclass"]}\nPC2', fontweight='bold')
        basis = ' (basis)' if age == REF_AGE else ''
        ax.set_title(f'{age}  n={P["n_by_age"][age]}{basis}', fontsize=9)
        sns.despine(ax=ax)

    draw_key(axes[row][ncol - 1], P)

fig.suptitle('yoo25 IT NR series in the P21 PC basis — colour = two score channels '
             '(diverging score_C − score_A, mixed toward score_B); rescaled within each age',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'\nSaved {OUT_PDF}')
print('\nDone.')
