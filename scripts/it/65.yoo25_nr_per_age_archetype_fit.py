"""Fit archetypes per timepoint at hand-chosen NOCs, in the shared P21 PC basis.

Second half of the pair started by 64. Script 64 sweeps NOC 2..5 at every timepoint and reports
the metrics; the NOC is then chosen BY EYE from its figure and hard-coded below, because no
single criterion reproduces the four curated NOCs already in the repo (see 64's docstring).
This script fits at those choices and labels each age's vertices against the adult archetypes.

Everything is fit in the SHARED P21 basis persisted by 61.v2, so vertices stay comparable across
ages and overlay on the 61.v3 grid. L5IT is excluded, matching 64.

Vertex IDENTITY is matched, not assumed. scomics.utils.pcha sorts vertices by inner-PC1, so the
k-th vertex carries no archetype meaning. Each vertex is labelled by the mean 33-36 archetype
marker score (61.v2's score_A/B/C) of its nearest cells -- script 62's rule, with one deliberate
change: 62 REQUIRES a bijection, which is only valid at a fixed NOC. Here NOC varies by age, so
when two vertices claim the same adult archetype the higher-scoring one keeps it and the other is
labelled `unmatched_<i>`. That is a RESULT, not an error -- at NOC=4 or 5 an unmatched vertex is
precisely the thing worth seeing -- so this script does not fail on it. The full mean-score matrix
is persisted so every assignment is auditable.

Caveats (inherited from 64, restated because this script's outputs are what gets read):
  - Cell count and age are confounded (L2/3 falls 6612 -> 2213 from P6 to P21). Quote any NOC
    trend from 64's matched_n pass, not from the `all` pass.
  - Every age is fit in the P21 basis, so an early age whose dominant variation is not captured by
    adult PCs may look lower-dimensional than it is. That is the price of comparability.
  - Vertices are fit in 5-D and drawn in 2-D: the simplex edges on the figure are projections and
    do not bound the 2-D cloud.
  - No batch correction anywhere; each age is its own set of replicates.
  - py_pcha needs PYTHONNOUSERSITE=1 (user-site numpy 2.2.6 shadows the env's pinned 1.26.4).

Reads:
  local_data/res/it/61.v2.yoo25_nr_<token>_pc_coords.tsv
  local_data/res/it/61.v2.yoo25_nr_<token>_archetype_scores.tsv
  local_data/res/it/62.yoo25_p21_<token>_archetype_coords.tsv     (P21 reference + PC orientation)
  local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv
Outputs:
  local_data/res/it/65.yoo25_nr_per_age_archetype_coords.tsv
  local_data/fig/it/65.yoo25_nr_per_age_archetype_fit.pdf
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from scomics.main import SCA

# --- file paths ---
RES_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it')
IT_EVO_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'it_evo')
FIG_DIR    = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'it')
IN_ARCMAP  = os.path.join(IT_EVO_DIR, '15.mouse_IT_joint_archetype_arc_order.tsv')
IN_COORDS  = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_pc_coords.tsv')
IN_SCORES  = os.path.join(RES_DIR, '61.v2.yoo25_nr_{token}_archetype_scores.tsv')
IN_P21_ARCH = os.path.join(RES_DIR, '62.yoo25_p21_{token}_archetype_coords.tsv')
OUT_COORDS = os.path.join(RES_DIR, '65.yoo25_nr_per_age_archetype_coords.tsv')
OUT_PDF    = os.path.join(FIG_DIR, '65.yoo25_nr_per_age_archetype_fit.pdf')

SUBCLASSES = [
    dict(subclass='L2/3', token='L23'),
    dict(subclass='L4',   token='L4'),
    dict(subclass='L6IT', token='L6IT'),
]

# --- parameters ---
AGE_COL      = 'Age'
TYPE_COL     = 'Type'
AGES         = ['P6', 'P8', 'P10', 'P12', 'P14', 'P17', 'P21']
NDIM         = 5             # PC1..PC5, matching 64, 62 and 33-36
DROP_PCS     = []
NOC_MIN, NOC_MAX = 2, 5      # the range 64 swept; choices outside it are rejected
ARCH_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
ARCH_COLORS  = {"A'": 'C0', "B'": 'C1', "C'": 'C2', "D'": 'C3'}
SUPERFICIAL_LABEL = "A'"
N_TOP_CELLS  = 300           # nearest cells per vertex used to read its marker identity
TOP_FRAC_CAP = 0.25          # ... capped at a quarter of the age, to keep the label local
RANDOM_SEED  = 0
POINT_SIZE   = 2.5
CELL_COLOR   = '#dcdcdc'
P21_REF_COLOR = '#9a9a9a'    # the P21 simplex, drawn on every panel for reference
FIG_PANEL_W  = 2.5
FIG_PANEL_H  = 2.5
DPI          = 300

# --- NOC per (subclass, age): CHOSEN FROM 64's EV CURVES ---
# 64's stability metric turned out unusable for this: ARV carries a median relative error of 0.19
# (upper quartile 0.30) at NOC=3, and its three candidate rules agree in only 5 of 42 units, so
# `argmax effEV` wanders between 2 and 5 with no trend. The EV curves, by contrast, are smooth and
# consistent in every panel, so the NOC is read from the EV elbow alone.
#
# The elbow is unambiguous: the marginal EV gain entering NOC=3 is the LARGEST step in 42/42
# units, and the step into NOC=4 is smaller everywhere. Hence NOC=3 throughout -- which is also
# the adult NOC of all three subclasses in 33-36, making this figure a comparison of archetype
# POSITION across age at fixed number.
#
# What does vary with age is the SHARPNESS of that elbow, gain(NOC4)/gain(NOC3):
#   L2/3  0.25 0.39 0.61 0.44 0.67 0.71 0.63   (P6 -> P21)
#   L6IT  0.32 0.32 0.64 0.46 0.53 0.72 0.59
#   L4    0.62 0.66 0.80 0.80 0.48 0.63 0.55   (no clean trend)
# Young L2/3 and L6IT have a crisp 3-archetype structure; by P21 a 4th archetype earns nearly as
# much as the 3rd. That is a statement about elbow sharpness, not about the chosen NOC.
#
# Override any cell to refit that timepoint at a different NOC; None stops the script.
NOC_BY_AGE = {
    'L2/3': {'P6': 3, 'P8': 3, 'P10': 3, 'P12': 3, 'P14': 3, 'P17': 3, 'P21': 3},
    'L4':   {'P6': 3, 'P8': 3, 'P10': 3, 'P12': 3, 'P14': 3, 'P17': 3, 'P21': 3},
    'L6IT': {'P6': 3, 'P8': 3, 'P10': 3, 'P12': 3, 'P14': 3, 'P17': 3, 'P21': 3},
}


def load_relabel():
    """{token: {old_letter: new_letter}} from the persisted depth arc (read, never hard-coded)."""
    arc = pd.read_csv(IN_ARCMAP, sep='\t')
    missing = {'token', 'old_letter', 'new_letter'} - set(arc.columns)
    if missing:
        raise ValueError(f'{IN_ARCMAP} missing column(s): {sorted(missing)}')
    return {token: dict(zip(sub['old_letter'], sub['new_letter']))
            for token, sub in arc.groupby('token')}


def match_vertices(aa_feat, xn, score_mat, letters, n_top):
    """Label each vertex by the adult archetype its nearest cells score highest on.

    Returns (labels, primed_ok, mean_scores). Unlike script 62 this tolerates a non-bijection:
    when two vertices claim one archetype the higher-scoring vertex keeps it and the other is
    `unmatched_<i>`, because at NOC > the adult NOC an extra vertex is a result, not an error.
    """
    noc = aa_feat.shape[0]
    dists = np.stack([np.linalg.norm(xn - aa_feat[k], axis=1) for k in range(noc)], axis=1)
    mean_scores = np.stack([score_mat[np.argsort(dists[:, k])[:n_top]].mean(axis=0)
                            for k in range(noc)])                   # (noc vertices, n letters)

    labels = [None] * noc
    claim = mean_scores.argmax(axis=1)
    for li, L in enumerate(letters):
        claimants = [k for k in range(noc) if claim[k] == li]
        if not claimants:
            continue
        winner = max(claimants, key=lambda k: mean_scores[k, li])
        labels[winner] = L
    for k in range(noc):
        if labels[k] is None:
            labels[k] = f'unmatched_{k}'
    return labels, mean_scores


print('--- per-timepoint archetype fits at the hand-chosen NOCs (script 64 -> 65) ---')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# fail fast on an unfilled or out-of-range NOC table, before any compute
unset = [(S, age) for S in NOC_BY_AGE for age in AGES if NOC_BY_AGE[S].get(age) is None]
if unset:
    raise ValueError(
        f'NOC_BY_AGE has {len(unset)} unset entry/entries, e.g. {unset[:5]}. Read '
        f'local_data/fig/it/64.yoo25_nr_per_age_noc_sweep.pdf and fill them in by hand -- '
        f'64 reports the sweep but deliberately does not choose.')
bad = [(S, age, NOC_BY_AGE[S][age]) for S in NOC_BY_AGE for age in AGES
       if not (NOC_MIN <= NOC_BY_AGE[S][age] <= NOC_MAX)]
if bad:
    raise ValueError(f'NOC outside the swept range {NOC_MIN}-{NOC_MAX}: {bad}')
for cfg in SUBCLASSES:
    if cfg['subclass'] not in NOC_BY_AGE:
        raise ValueError(f'NOC_BY_AGE has no entry for {cfg["subclass"]}')
    for tmpl in (IN_COORDS, IN_SCORES, IN_P21_ARCH):
        path = tmpl.format(token=cfg['token'])
        if not os.path.exists(path):
            raise FileNotFoundError(f'{cfg["subclass"]}: missing {path} -- run 61.v2 then 62 first')
if not os.path.exists(IN_ARCMAP):
    raise FileNotFoundError(f'missing depth-arc table {IN_ARCMAP}')

relabel_by_token = load_relabel()
np.random.seed(RANDOM_SEED)                   # PCHA initialises randomly
pc_cols = [f'PC{i+1}' for i in range(NDIM)]
out_rows, panels = [], []

for cfg in SUBCLASSES:
    S, token = cfg['subclass'], cfg['token']
    print(f'\n=== {S} ===')
    coords = pd.read_csv(IN_COORDS.format(token=token), sep='\t', index_col=0)
    scores = pd.read_csv(IN_SCORES.format(token=token), sep='\t', index_col=0).reindex(coords.index)
    if scores.isna().any().any():
        raise ValueError(f'{S}: cells missing from {IN_SCORES.format(token=token)}')
    letters = [L for L in ARCH_LETTERS if f'score_{L}' in scores.columns]
    score_cols = [f'score_{L}' for L in letters]
    relabel = relabel_by_token[token]
    if set(relabel) != set(letters):
        raise ValueError(f'{S}: arc-table letters {sorted(relabel)} != score letters {letters}')

    # --- PC orientation: the 61.v3 rule, from 62's P21 vertices, so panels line up with it ---
    p21 = pd.read_csv(IN_P21_ARCH.format(token=token), sep='\t', index_col=0)
    if list(p21.index) != letters:
        raise ValueError(f'{S}: {IN_P21_ARCH.format(token=token)} index {list(p21.index)} '
                         f'!= score letters {letters}')
    primed = [relabel[L] for L in letters]
    if SUPERFICIAL_LABEL not in primed:
        raise ValueError(f'{S}: no {SUPERFICIAL_LABEL} among {primed}')
    ia = primed.index(SUPERFICIAL_LABEL)
    io = [i for i in range(len(letters)) if i != ia][-1]          # the opposite flank
    p21_xy = p21[['PC1', 'PC2']].values
    sx = -1.0 if p21_xy[ia, 0] > p21_xy[io, 0] else 1.0
    if 'B' in letters and relabel['B'] not in (SUPERFICIAL_LABEL,):
        ib = letters.index('B')
        flanks = [i for i in range(len(letters)) if i != ib]
        sy = -1.0 if p21_xy[ib, 1] > p21_xy[flanks, 1].mean() else 1.0
    else:
        sy = 1.0
    sign = np.array([sx, sy])
    print(f'  PC orientation from 62: PC1 x{sx:+.0f}, PC2 x{sy:+.0f}')

    ages_all = coords[AGE_COL].astype(str).values
    per_age = {}
    for age in AGES:
        m = ages_all == age
        if not m.any():
            raise ValueError(f'{S}: no cells at {age}')
        sub = coords[m]
        xn = sub[pc_cols].values
        noc = NOC_BY_AGE[S][age]

        sca = SCA(xn, sub[TYPE_COL].astype(str).values)
        sca.setup_feature_matrix(method='data')
        _, aa, varexpl = sca.proj_and_pcha(NDIM, noc, drop_pcs=DROP_PCS)
        # back-project out of the inner PCA into the PC1..PC5 feature space (33-36 / 62 idiom)
        aa_feat = sca.aa.T @ sca.pca_.components_[:NDIM] + sca.pca_.mean_      # (noc, NDIM)

        n_top = int(min(N_TOP_CELLS, max(noc * 5, TOP_FRAC_CAP * m.sum())))
        labels, mean_scores = match_vertices(
            aa_feat, xn, scores.loc[sub.index, score_cols].values, letters, n_top)
        n_unmatched = sum(str(l).startswith('unmatched') for l in labels)
        print(f'  {age:4s} n={int(m.sum()):5d} NOC={noc}  varexpl={varexpl:.3f}  '
              f'labels={[(l if not str(l).startswith("unmatched") else "-") for l in labels]}'
              f'{"  (" + str(n_unmatched) + " unmatched)" if n_unmatched else ""}')

        for k in range(noc):
            L = labels[k]
            out_rows.append(dict(
                subclass=S, age=age, noc=noc, vertex=k, matched_letter=L,
                primed=relabel[L] if L in relabel else '',
                varexpl=varexpl, n_cells=int(m.sum()), n_top_cells=n_top,
                **{c: aa_feat[k, i] for i, c in enumerate(pc_cols)},
                **{f'mean_{c}': mean_scores[k, i] for i, c in enumerate(score_cols)}))

        per_age[age] = dict(xy=xn[:, :2] * sign, vx=aa_feat[:, :2] * sign,
                            labels=labels, relabel=relabel, n=int(m.sum()), noc=noc)

    panels.append(dict(subclass=S, per_age=per_age, p21_ref=p21_xy * sign, sx=sx, sy=sy))

out = pd.DataFrame(out_rows)
out.to_csv(OUT_COORDS, sep='\t', index=False)
print(f'\nSaved {OUT_COORDS}')

# ===================== figure: subclass rows x age cols =====================

plt.rcParams['pdf.fonttype'] = 42             # editable vector text
fig, axes = plt.subplots(len(panels), len(AGES), squeeze=False,
                         figsize=(FIG_PANEL_W * len(AGES), FIG_PANEL_H * len(panels)))

for row, P in enumerate(panels):
    allxy = np.vstack([P['per_age'][a]['xy'] for a in AGES]
                      + [P['per_age'][a]['vx'] for a in AGES] + [P['p21_ref']])
    pad = 0.04 * (allxy.max(axis=0) - allxy.min(axis=0))
    xlim = (allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
    ylim = (allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])

    for col, age in enumerate(AGES):
        ax = axes[row][col]
        A = P['per_age'][age]
        ax.scatter(A['xy'][:, 0], A['xy'][:, 1], s=POINT_SIZE, c=CELL_COLOR,
                   linewidths=0, zorder=1, rasterized=True)

        # P21 reference simplex, same on every panel, so the drift of the geometry is readable
        ref = P['p21_ref']
        for i, j in itertools.combinations(range(len(ref)), 2):
            ax.plot(ref[[i, j], 0], ref[[i, j], 1], '--', color=P21_REF_COLOR,
                    linewidth=0.8, zorder=2)

        # this age's own simplex: all pairwise edges, which is the honest 2-D view of a
        # k-vertex simplex (for NOC=3 it is the triangle, for NOC=2 a single segment)
        vx = A['vx']
        for i, j in itertools.combinations(range(len(vx)), 2):
            ax.plot(vx[[i, j], 0], vx[[i, j], 1], '-', color='black', linewidth=0.9, zorder=3)
        ax.scatter(vx[:, 0], vx[:, 1], marker='D', c='black', s=36, zorder=4)
        for (vx_, vy_), L in zip(vx, A['labels']):
            txt = A['relabel'][L] if L in A['relabel'] else '?'
            ax.annotate(txt, (vx_, vy_), textcoords='offset points', xytext=(4, 4),
                        fontsize=7, fontweight='bold', color='black', zorder=5)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            pc1 = 'PC1' if P['sx'] > 0 else '−PC1'
            pc2 = 'PC2' if P['sy'] > 0 else '−PC2'
            ax.set_ylabel(f'{P["subclass"]}\n[{pc1} , {pc2}]', fontweight='bold', fontsize=8)
        ax.set_title(f'{age}  n={A["n"]}  NOC={A["noc"]}', fontsize=9)
        sns.despine(ax=ax)

fig.suptitle('Per-timepoint archetype fits in the shared P21 PC basis '
             '(black = that age\'s simplex, dashed grey = the P21 simplex of script 62)',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(OUT_PDF, bbox_inches='tight', dpi=DPI)
plt.close(fig)
print(f'Saved {OUT_PDF}')
print('\nDone.')
