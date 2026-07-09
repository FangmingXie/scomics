"""Spearman conservation heatmap (mouse ABC vs human ABCD), vectorized PDF.

Reproduces the Spearman-correlation panel from
26.mouse_human_archetype_conservation.py using seaborn's sns.heatmap
with a rocket_r colorbar, saved as an editable vector PDF.

Reads:
  local_data/res/l23_evo/26.archetype_spearman.tsv
Outputs:
  local_data/fig/l23_evo/26.spearman_heatmap.pdf
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- file paths ---
OUT_FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'l23_evo')
IN_CORR     = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'l23_evo', '26.archetype_spearman.tsv')
OUT_PDF     = os.path.join(OUT_FIG_DIR, '26.spearman_heatmap.pdf')

os.makedirs(OUT_FIG_DIR, exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42   # editable vector text

# --- load Spearman correlation matrix (mouse rows x human cols) ---
corr = pd.read_csv(IN_CORR, sep='\t', index_col=0)
corr.index   = [i.replace('mouse_', '') for i in corr.index]
corr.columns = [c.replace('human_', '') for c in corr.columns]

# reverse human order (D, C, B, A) and relabel as A', B', C', D'
corr = corr[corr.columns[::-1]]
corr.columns = [f"{chr(ord('A') + i)}'" for i in range(corr.shape[1])]

# --- heatmap ---
fig, ax = plt.subplots(figsize=(4.2, 3.2))
sns.heatmap(
    corr, ax=ax,
    cmap='RdBu_r', vmin=-0.2, vmax=0.2, center=0,
    annot=True, fmt='.2f', annot_kws={'size': 9},
    linewidths=0.5, linecolor='white', square=True,
    cbar_kws={'label': 'Spearman r', 'ticks': [-0.2, 0, 0.2]},
)
ax.set_xlabel('Human archetype')
ax.set_ylabel('Mouse archetype')
ax.set_title('Mouse (ABC) vs Human (ABCD)\narchetype conservation')
ax.tick_params(length=0)

fig.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
