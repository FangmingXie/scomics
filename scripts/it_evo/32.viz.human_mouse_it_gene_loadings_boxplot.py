"""Mouse-vs-human ortholog gene-loading joint plots on the conserved CCA axes, all four IT layers.

Extends `24b.viz.human_mouse_L23_gene_loadings_boxplot.py` (L2/3 only) to the IT subclasses
(L2/3, L4, L5IT, L6IT). One figure, an N x 2 grid of joint plots: rows are subclasses, columns
are the two conserved canonical axes (CCA1, CCA2). Each cell is the same joint plot as 24b —
scatter of per-gene mouse-vs-human canonical projections coloured by mouse archetype, with a
horizontal boxplot per group below the x-axis and a vertical one along the y-axis, plus
Mann-Whitney U stars for ALL pairwise comparisons among the groups. See 24b for why boxes
beat KDE marginals.

Pairwise annotation (both marginals, per species). Every pair of the four boxplot groups
(the gray "other" bulk plus each mouse archetype) is tested, in two visual forms because they
answer different questions and a uniform bracket set would need ~5 stacked lanes:
  - archetype vs archetype -> a bracket spanning the two slots, packed shortest-span-first
    into as few lanes as fit without touching;
  - archetype vs the "other" bulk -> a star in its own outer lane, as before, since that is
    the "which pole does this archetype sit on" comparison and every archetype shares the
    same reference group.
Both live in a band reserved past the data by widening the shared axis limits, so no
annotation overlaps a box. Full p-values and rank-biserial effect sizes for every pair go to
stdout. As in 24b these are descriptive: the groups are not independent and NO multiplicity
correction is applied — with all pairs now shown that is a larger family, so read the stars
alongside the rank-biserial sizes rather than as a family-wise test.

A subclass is drawn only if every input it needs for the chosen universe exists (else it is
logged and omitted). Under hvg_intersect all four layers qualify; under hvg_union only L2/3 and
L4 do, because 26's out-of-sample loading extension fails its reconstruction check for the L5IT
and L6IT mouse loadings (it/23, it/25) — those two layers are dropped from the union figure
rather than drawn from unverified loadings.

Per-layer, per-species the archetype relabel/reorder is READ from the persisted mouse depth arc
`15.mouse_IT_joint_archetype_arc_order.tsv` (the same relabel scripts 12-14/18c use), so the
figure letters match the rest of the project: internal 05 letters A/B/C are shown as the primed
figure letters (e.g. L2/3 A->C', but L6IT A->A'), ordered A', B', C', and each archetype's colour
follows its DISPLAYED label (A'->C0, B'->C1, C'->C2). Only the figure is relabelled; the score
TSVs and markers keep 05's letters.

--universe selects the gene set (mirrors 16/24b): hvg_intersect | hvg_union (default). Because
the canonical correlation depends on the gene universe, r is NOT comparable across universes or
across layers — each cell annotates its universe and n. Per script 21 only L2/3 has clearly
conserved axes; L4/L5IT/L6IT correlations are weak and their panels are exploratory.

Reads (per subclass TOKEN; paths switch with UNIVERSE):
  local_data/res/it_evo/02.human_<TOKEN>_varimax_loadings.tsv            (HVG membership)
  local_data/res/it/{19,21,23,25}.cheng22_<TOKEN>_varimax_loadings.tsv   (HVG membership)
  local_data/res/it_evo/26.{human,mouse}_<TOKEN>_varimax_loadings_full.tsv   (hvg_union only)
  local_data/res/it_evo/16.<TOKEN>_axis_cca_weights_{human,mouse}<SUFFIX>.tsv
  local_data/res/it_evo/05.mouse_<TOKEN>_archetype_markers.tsv
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv        (figure relabel)
  data/human_mouse_orthologs.tsv
Outputs (all four layers x CCA1/CCA2 in one figure):
  local_data/fig/it_evo/32.human_mouse_it_gene_loadings<SUFFIX>.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- config ---
UNIVERSE_SUFFIX = {'hvg_intersect': '', 'hvg_union': '_union'}
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--universe', choices=list(UNIVERSE_SUFFIX), default='hvg_union')
UNIVERSE = _parser.parse_args().universe
SUFFIX   = UNIVERSE_SUFFIX[UNIVERSE]

# Gate-A VX sets mirror 16's SUBCLASSES; mouse_loadings is the it/{19,21,23,25} HVG file.
SUBCLASSES = [
    {'token': 'L23',  'label': 'L2/3',
     'human_vx': ['VX2', 'VX6', 'VX7', 'VX8', 'VX9', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX7', 'VX8', 'VX9'],
     'mouse_loadings': '19.cheng22_L23_varimax_loadings.tsv'},
    {'token': 'L4',   'label': 'L4',
     'human_vx': ['VX3', 'VX4', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX4', 'VX5', 'VX6'],
     'mouse_loadings': '21.cheng22_L4_varimax_loadings.tsv'},
    {'token': 'L5IT', 'label': 'L5IT',
     'human_vx': ['VX3', 'VX5', 'VX6', 'VX8', 'VX10'],
     'mouse_vx': ['VX1', 'VX2', 'VX5', 'VX6', 'VX7'],
     'mouse_loadings': '23.cheng22_L5IT_varimax_loadings.tsv'},
    {'token': 'L6IT', 'label': 'L6IT',
     'human_vx': ['VX6', 'VX7', 'VX8', 'VX9'],
     'mouse_vx': ['VX1', 'VX2', 'VX9', 'VX10'],
     'mouse_loadings': '25.cheng22_L6IT_varimax_loadings.tsv'},
]
AXES         = ['CCA1', 'CCA2']
ALPHABET     = ['A', 'B', 'C', 'D', 'E', 'F']
# colour follows the DISPLAYED (primed) figure label: A'->C0, B'->C1, C'->C2, ...
DISPLAY_COLOR = {f"{L}'": f'C{i}' for i, L in enumerate(ALPHABET)}
BASE_COLOR   = '#bdbdbd'                            # "other" genes: neutral gray
POINT_SIZE   = {'hvg_intersect': 10, 'hvg_union': 4}[UNIVERSE]
BASE_ALPHA   = {'hvg_intersect': 1.0, 'hvg_union': 0.45}[UNIVERSE]
ARCH_BUMP    = {'hvg_intersect': 12, 'hvg_union': 6}[UNIVERSE]
TOP_N_LABEL  = 15                                   # label the top genes by |mouse.human| loading
BOX_WIDTH    = 0.62                                 # fraction of the unit slot each box fills
FLIER_SIZE   = 3.0                                  # outlier marker size (points)
SIG_LEVELS   = [(0.001, '***'), (0.01, '**'), (0.05, '*')]   # MWU; else 'ns'
PAIR_LANE_FRAC = 0.06     # value-axis fraction reserved per archetype-pair bracket lane
BRACKET_LW     = 0.7      # bracket line width
BRACKET_TICK   = 0.20     # bracket end-tick length, as a fraction of one lane's width
N_PERM       = 20000                                # gene-label permutations (mirror 24b)
PERM_SEED    = 0                                    # mirror 24b's seed
R_MATCH_TOL  = 1e-6                                 # panel r vs canonical r (see prep_subclass)

# --- file paths ---
RES_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
IT_RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
FIG_DIR       = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it_evo')
IN_ARC        = os.path.join(RES_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
IN_ORTHOLOGS  = os.path.join(PROJECT_ROOT, 'data', 'human_mouse_orthologs.tsv')
OUT_PDF       = os.path.join(FIG_DIR, f'32.human_mouse_it_gene_loadings{SUFFIX}.pdf')

os.makedirs(FIG_DIR, exist_ok=True)

# --- shared inputs ---
ORTHO = (pd.read_csv(IN_ORTHOLOGS, sep='\t')
         .drop_duplicates('human_symbol').drop_duplicates('mouse_symbol'))
ARC = pd.read_csv(IN_ARC, sep='\t')


def relabel_for(token):
    """{old 05 letter A/B/C: primed figure letter} for one subclass, from 15's depth arc."""
    sub = ARC[ARC['token'] == token]
    return dict(zip(sub['old_letter'], sub['new_letter']))


def paths_for(cfg):
    """Input paths this subclass needs for the current UNIVERSE (used for skip-check and load)."""
    token = cfg['token']
    p = {'h_hvg':   os.path.join(RES_DIR, f'02.human_{token}_varimax_loadings.tsv'),
         'm_hvg':   os.path.join(IT_RES_DIR, cfg['mouse_loadings']),
         'w_h':     os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_human{SUFFIX}.tsv'),
         'w_m':     os.path.join(RES_DIR, f'16.{token}_axis_cca_weights_mouse{SUFFIX}.tsv'),
         'markers': os.path.join(RES_DIR, f'05.mouse_{token}_archetype_markers.tsv')}
    if UNIVERSE == 'hvg_union':   # the expanded universe reads 26's out-of-sample loadings
        p['h_full'] = os.path.join(RES_DIR, f'26.human_{token}_varimax_loadings_full.tsv')
        p['m_full'] = os.path.join(RES_DIR, f'26.mouse_{token}_varimax_loadings_full.tsv')
    return p


def _orthonormal(Mtx):
    return np.linalg.qr(Mtx - Mtx.mean(axis=0))[0]


def permutation_pvals(X, Y):
    """Cross-species canonical correlations + gene-label permutation p per axis (mirrors 24b)."""
    Qx, Qy = _orthonormal(X), _orthonormal(Y)
    n = X.shape[0]
    obs = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)
    rng = np.random.default_rng(PERM_SEED)
    null = np.empty((N_PERM, min(Qx.shape[1], Qy.shape[1])))
    for i in range(N_PERM):
        null[i] = np.clip(np.linalg.svd(Qx.T @ Qy[rng.permutation(n)], compute_uv=False), 0.0, 1.0)
    out = {}
    for comp, axis in enumerate(AXES):
        n_ge = int((null[:, comp] >= obs[comp]).sum())
        out[axis] = {'r_cca': float(obs[comp]), 'n_ge': n_ge, 'p': (1 + n_ge) / (N_PERM + 1)}
    return out


def prep_subclass(cfg):
    """All per-subclass quantities the two cells (CCA1, CCA2) need. Mirrors 24b's preamble."""
    token, hvx, mvx = cfg['token'], cfg['human_vx'], cfg['mouse_vx']
    P = paths_for(cfg)
    hvg_h = pd.read_csv(P['h_hvg'], sep='\t', index_col=0)
    hvg_m = pd.read_csv(P['m_hvg'], sep='\t', index_col=0)
    if UNIVERSE == 'hvg_intersect':
        H, M = hvg_h, hvg_m
    else:
        H = pd.read_csv(P['h_full'], sep='\t', index_col=0)
        M = pd.read_csv(P['m_full'], sep='\t', index_col=0)
    shared = ORTHO[ORTHO['human_symbol'].isin(H.index) & ORTHO['mouse_symbol'].isin(M.index)]
    if UNIVERSE == 'hvg_union':
        shared = shared[shared['human_symbol'].isin(hvg_h.index)
                        | shared['mouse_symbol'].isin(hvg_m.index)]
    shared = shared.reset_index(drop=True)

    X = H.loc[shared['human_symbol'].values, hvx].values
    Y = M.loc[shared['mouse_symbol'].values, mvx].values
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)

    wdf_h = pd.read_csv(P['w_h'], sep='\t', index_col=0)
    wdf_m = pd.read_csv(P['w_m'], sep='\t', index_col=0)

    def cca_load(axis):
        # .loc row mixes floats + the bool `stable` column, so force float before the matmul.
        wm = wdf_m.loc[axis, mvx].to_numpy(dtype=float)
        wh = wdf_h.loc[axis, hvx].to_numpy(dtype=float)
        return Yc @ wm, Xc @ wh

    # mouse archetype membership per gene (highest-log2FC archetype if multiple)
    mk = pd.read_csv(P['markers'], sep='\t')
    mk['letter'] = mk['archetype'].map(lambda a: ALPHABET[int(a.split('_')[1]) - 1])
    mk = mk.sort_values('log2FC', ascending=False).drop_duplicates('gene')
    gene2arch = dict(zip(mk['gene'], mk['letter']))
    letters = np.array([gene2arch.get(g, '') for g in shared['mouse_symbol'].values])

    relabel = relabel_for(token)                       # old letter -> primed figure letter
    order = sorted(relabel, key=lambda L: relabel[L])  # internal keys ordered so labels read A',B',C'
    groups = [('other', BASE_COLOR)] + [(L, DISPLAY_COLOR[relabel[L]]) for L in order]

    return {'label': cfg['label'], 'symbols': shared['human_symbol'].values, 'n': len(shared),
            'letters': letters, 'relabel': relabel, 'order': order, 'groups': groups,
            'cca_load': cca_load, 'sig': permutation_pvals(X, Y), 'n_markers': len(gene2arch)}


# --- per-group helpers (all keyed by internal letter; S carries the per-subclass context) ---
def group_values(vals, S):
    letters = S['letters']
    return {'other': vals[letters == ''], **{L: vals[letters == L] for L in S['order']}}


def mwu(a, b):
    """Two-sided Mann-Whitney U of a vs b -> (p, rank-biserial). +1 = a entirely above b."""
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    return p, 2 * u / (len(a) * len(b)) - 1


def mwu_vs_other(vals, S):
    """Mann-Whitney U of each archetype vs the 'other' bulk. internal letter -> (p, rank-biserial)."""
    g = group_values(vals, S)
    return {L: mwu(g[L], g['other']) for L in S['order'] if len(g[L]) >= 1}


def mwu_arch_pairs(vals, S):
    """Mann-Whitney U of every archetype-vs-archetype pair. (letter_a, letter_b) -> (p, rb).

    Together with mwu_vs_other these cover ALL pairs among the four boxplot groups
    (the "other" bulk plus each archetype). Both are uncorrected for multiplicity — see
    the module docstring; they are descriptive, not a family-wise test.
    """
    g = group_values(vals, S)
    return {(a, b): mwu(g[a], g[b]) for a, b in combinations(S['order'], 2)
            if len(g[a]) >= 1 and len(g[b]) >= 1}


def archetype_pair_lanes(S):
    """Bracket layout for the archetype-vs-archetype pairs: [(lo, hi, a, b, lane)], n_lanes.

    Slots are the group positions in S['groups']. Pairs are packed greedily shortest-span
    first, a pair joining a new lane only when its span neither overlaps NOR touches one
    already in that lane (touching would collide the end ticks).
    """
    pos = {g: k for k, (g, _) in enumerate(S['groups'])}
    pairs = sorted(((min(pos[a], pos[b]), max(pos[a], pos[b]), a, b)
                    for a, b in combinations(S['order'], 2)),
                   key=lambda t: (t[1] - t[0], t[0]))
    lanes, out = [], []
    for lo, hi, a, b in pairs:
        for k, occupied in enumerate(lanes):
            if all(hi < o_lo or lo > o_hi for o_lo, o_hi in occupied):
                occupied.append((lo, hi))
                out.append((lo, hi, a, b, k))
                break
        else:
            lanes.append([(lo, hi)])
            out.append((lo, hi, a, b, len(lanes) - 1))
    return out, len(lanes)


def stars(p):
    for thresh, mark in SIG_LEVELS:
        if p < thresh:
            return mark
    return 'ns'


def report_mwu(axis_label, ms, hs, S):
    """Print every pairwise MWU: each archetype vs the bulk, then archetype vs archetype."""
    for species, vals in (('mouse', ms), ('human', hs)):
        g = group_values(vals, S)
        vs_other = mwu_vs_other(vals, S)
        parts = [f'{S["relabel"][L]}: med {np.median(g[L]):+.3f} '
                 f'rb {vs_other[L][1]:+.2f} p {vs_other[L][0]:.1e} {stars(vs_other[L][0])}'
                 for L in S['order'] if L in vs_other]
        print(f'    {S["label"]:5s} {axis_label} {species:5s} vs other '
              f'(bulk med {np.median(g["other"]):+.3f})  ' + ' | '.join(parts))
        pairs = [f'{S["relabel"][a]} vs {S["relabel"][b]}: rb {rb:+.2f} p {p:.1e} {stars(p)}'
                 for (a, b), (p, rb) in mwu_arch_pairs(vals, S).items()]
        if pairs:
            print(f'    {S["label"]:5s} {axis_label} {species:5s} pairs                     '
                  + '  | '.join(pairs))


def draw_boxes(ax, values_by_group, orientation, S):
    """One box per non-empty group at its slot position, coloured to match the scatter."""
    data, positions, colors = [], [], []
    for pos, (g, color) in enumerate(S['groups']):
        v = values_by_group[g]
        if len(v) == 0:
            continue                       # a layer's tiny intersect set can leave an archetype empty
        data.append(v); positions.append(pos); colors.append(color)
    bp = ax.boxplot(data, positions=positions, widths=BOX_WIDTH, orientation=orientation,
                    patch_artist=True, showfliers=True, medianprops=dict(color='black', lw=1.2),
                    whiskerprops=dict(color='0.4', lw=0.8), capprops=dict(color='0.4', lw=0.8),
                    flierprops=dict(marker='o', markersize=FLIER_SIZE, markerfacecolor='0.5',
                                    markeredgecolor='none', alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.55)
        patch.set_edgecolor('0.35'); patch.set_linewidth(0.8)


def annotate_sig(ax, vals, orientation, S, v):
    """Significance stars vs the 'other' bulk, in their own lane at value-axis position `v`."""
    positions = [g for g, _ in S['groups']]
    for L, (p, _) in mwu_vs_other(vals, S).items():
        pos = positions.index(L)
        if orientation == 'horizontal':
            ax.text(v, pos, stars(p), ha='center', va='center', fontsize=8, color='0.25')
        else:
            ax.text(pos, v, stars(p), ha='center', va='center', fontsize=8, color='0.25')


def annotate_pairs(ax, vals, orientation, S, pair_lanes, data_max, lane_w):
    """Archetype-vs-archetype MWU stars as brackets in the reserved band past the data.

    The band beyond `data_max` is created by padding `lim` in draw_joint, so the brackets
    never overlap the boxes and the marginal stays aligned with the shared scatter axis.
    """
    g = group_values(vals, S)
    tick = BRACKET_TICK * lane_w
    for lo, hi, a, b, k in pair_lanes:
        if len(g[a]) < 1 or len(g[b]) < 1:
            continue
        p, _ = mwu(g[a], g[b])
        v = data_max + (k + 0.5) * lane_w
        if orientation == 'horizontal':          # groups on y, values on x
            ax.plot([v, v], [lo, hi], color='0.35', lw=BRACKET_LW)
            for slot in (lo, hi):
                ax.plot([v - tick, v], [slot, slot], color='0.35', lw=BRACKET_LW)
            ax.text(v + tick, (lo + hi) / 2, stars(p), rotation=90,
                    ha='center', va='center', fontsize=7, color='0.25')
        else:                                    # groups on x, values on y
            ax.plot([lo, hi], [v, v], color='0.35', lw=BRACKET_LW)
            for slot in (lo, hi):
                ax.plot([slot, slot], [v - tick, v], color='0.35', lw=BRACKET_LW)
            ax.text((lo + hi) / 2, v + tick, stars(p),
                    ha='center', va='bottom', fontsize=7, color='0.25')


def draw_joint(subfig, S, axis_label, ms, hs, sig):
    """Draw one (subclass, axis) joint plot into `subfig`; returns the panel Pearson r."""
    r = float(np.corrcoef(ms, hs)[0, 1])
    if not abs(abs(r) - sig['r_cca']) < R_MATCH_TOL:
        raise ValueError(
            f'{S["label"]} {axis_label}: projected Pearson |r| = {abs(r):.6f} does not match the '
            f'canonical correlation {sig["r_cca"]:.6f} the permutation p-value tests — 16\'s saved '
            f'weights are not the canonical vectors for this gene universe')
    p_str = (f'p < {1 / (N_PERM + 1):.0e}' if sig['n_ge'] == 0 else f'p = {sig["p"]:.1e}')
    top_idx = np.argsort(np.abs(ms * hs))[::-1][:TOP_N_LABEL]
    data_lim = np.array([min(ms.min(), hs.min()), max(ms.max(), hs.max())]) * 1.08

    # Reserve a band past the data on the value axis for the significance annotations: one
    # lane per archetype-pair bracket, plus a final lane for the vs-"other" stars. Both
    # marginals share their value axis with the scatter, so the band widens `lim` for all
    # three axes — the scatter simply gains matching headroom on both axes and stays square.
    pair_lanes, n_lanes = archetype_pair_lanes(S)
    lane_w = PAIR_LANE_FRAC * (data_lim[1] - data_lim[0])
    lim = np.array([data_lim[0], data_lim[1] + (n_lanes + 1) * lane_w])
    v_other = data_lim[1] + (n_lanes + 0.5) * lane_w   # vs-"other" star lane

    gs = subfig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],
                             wspace=0.04, hspace=0.04)
    ax_main = subfig.add_subplot(gs[0, 0])
    ax_boxx = subfig.add_subplot(gs[1, 0], sharex=ax_main)   # below x-axis
    ax_boxy = subfig.add_subplot(gs[0, 1], sharey=ax_main)   # along y-axis

    letters = S['letters']
    n_other = int((letters == '').sum())
    ax_main.scatter(ms[letters == ''], hs[letters == ''], s=POINT_SIZE, c=BASE_COLOR,
                    linewidths=0, alpha=BASE_ALPHA, rasterized=True, label=f'other (n={n_other})')
    for L in S['order']:
        m = letters == L
        ax_main.scatter(ms[m], hs[m], s=POINT_SIZE + ARCH_BUMP, c=DISPLAY_COLOR[S['relabel'][L]],
                        linewidths=0, alpha=0.9, zorder=3,
                        label=f'mouse archetype {S["relabel"][L]} (n={int(m.sum())})')

    ax_main.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax_main.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax_main.plot(data_lim, data_lim, '--', color='0.6', lw=0.8, zorder=0)
    ax_main.scatter(ms[top_idx], hs[top_idx], s=POINT_SIZE + ARCH_BUMP + 6, facecolors='none',
                    edgecolors='black', linewidths=0.7, zorder=4)
    for i in top_idx:
        ax_main.annotate(S['symbols'][i], (ms[i], hs[i]), textcoords='offset points',
                         xytext=(4, 3), fontsize=6, fontstyle='italic', zorder=5)

    ax_main.set_xlim(lim); ax_main.set_ylim(lim)
    ax_main.set_ylabel(f'Human Jorstad23 {axis_label} gene loading')
    ax_main.set_title(f'{S["label"]} {axis_label}: mouse vs human\n'
                      f'r = {r:.3f},  {p_str}  ({N_PERM} perms)\n'
                      f'universe: {UNIVERSE},  {S["n"]} orthologous genes', fontsize=9)
    ax_main.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax_main.tick_params(labelbottom=False)
    sns.despine(ax=ax_main)

    labels = ['other' if g == 'other' else S['relabel'][g] for g, _ in S['groups']]
    draw_boxes(ax_boxx, group_values(ms, S), 'horizontal', S)
    draw_boxes(ax_boxy, group_values(hs, S), 'vertical', S)
    annotate_sig(ax_boxx, ms, 'horizontal', S, v_other)
    annotate_sig(ax_boxy, hs, 'vertical', S, v_other)
    annotate_pairs(ax_boxx, ms, 'horizontal', S, pair_lanes, data_lim[1], lane_w)
    annotate_pairs(ax_boxy, hs, 'vertical', S, pair_lanes, data_lim[1], lane_w)

    ax_boxx.set_ylim(len(S['groups']) - 0.5, -0.5)   # group 0 ("other") nearest the scatter
    ax_boxx.set_yticks(range(len(S['groups'])))
    ax_boxx.set_yticklabels(labels, fontsize=8)
    ax_boxx.axvline(0, color='0.75', lw=0.6, zorder=0)
    ax_boxx.set_xlabel(f'Mouse Cheng22 {axis_label} gene loading')

    ax_boxy.set_xlim(-0.5, len(S['groups']) - 0.5)
    ax_boxy.set_xticks(range(len(S['groups'])))
    ax_boxy.set_xticklabels(labels, fontsize=8)
    ax_boxy.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax_boxy.tick_params(labelleft=False)
    ax_boxy.set_xlabel('MWU, all pairs: brackets = archetype pairs, outer stars = vs "other"\n'
                       'ns * .05 ** .01 *** .001 (uncorrected)', fontsize=6)

    sns.despine(ax=ax_boxx)
    sns.despine(ax=ax_boxy)
    return r


print(f'--- IT CCA gene-loading joint plots, all IT layers (boxplot marginals) ---')
print(f'  universe: {UNIVERSE}')

# A layer is drawn only if every input it needs for this universe exists. hvg_union reads 26's
# extended loadings, which exist only where 26's reconstruction check passed (L2/3, L4) — L5IT
# and L6IT mouse loadings fail it, so they are omitted from the union figure (logged, not faked).
active, skipped = [], []
for cfg in SUBCLASSES:
    missing = [os.path.basename(p) for p in paths_for(cfg).values() if not os.path.exists(p)]
    (skipped if missing else active).append((cfg, missing))
for cfg, missing in skipped:
    print(f'  SKIP {cfg["label"]:5s} ({UNIVERSE}): missing {missing}')
if not active:
    raise FileNotFoundError(f'no subclass has all inputs for universe {UNIVERSE}')
active = [cfg for cfg, _ in active]
print(f'  layers drawn: {[c["label"] for c in active]}')

plt.rcParams['pdf.fonttype'] = 42
fig = plt.figure(figsize=(16, 8 * len(active)))
subfigs = fig.subfigures(len(active), len(AXES), squeeze=False, wspace=0.06, hspace=0.10)

for i, cfg in enumerate(active):
    S = prep_subclass(cfg)
    print(f'  {S["label"]}: {S["n"]} shared orthologs; archetype genes '
          f'{ {S["relabel"][L]: int((S["letters"] == L).sum()) for L in S["order"]} } '
          f'of {S["n_markers"]} markers')
    for j, axis in enumerate(AXES):
        ms, hs = S['cca_load'](axis)
        report_mwu(axis, ms, hs, S)
        r = draw_joint(subfigs[i][j], S, axis, ms, hs, S['sig'][axis])
        print(f'    {S["label"]:5s} {axis}: r = {r:.3f}, '
              f'p = {S["sig"][axis]["p"]:.2e} ({S["sig"][axis]["n_ge"]}/{N_PERM} >= obs)')

fig.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved {OUT_PDF}')
print('\nDone.')
