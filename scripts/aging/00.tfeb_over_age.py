import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- file paths ---
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'links', 'aging', 'hahn23_vis_bulk.parquet')
FIG_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'fig', 'aging')
OUT_FIG = os.path.join(FIG_DIR, '00.tfeb_over_age.png')

# --- config ---
GENES = ['Tfeb', 'Cln3', 'Fermt1', 'Fst']
SEX_COLORS = {'Female': '#E07B8A', 'Male': '#5B9BD5'}

# === main ===
os.makedirs(FIG_DIR, exist_ok=True)

# --- load ---
df = pd.read_parquet(INPUT_FILE)

# --- normalize: log2(1+CPM) ---
cpm = df.div(df.sum(axis=0), axis=1) * 1e6
log_cpm = np.log2(1 + cpm)

# --- tidy expression for all genes ---
records = []
for col in log_cpm.columns:
    parts = col.split('_')
    age_mo = int(parts[0].replace('mo', ''))
    sex = parts[1]
    for gene in GENES:
        records.append({'age_mo': age_mo, 'sex': sex, 'gene': gene, 'expr': log_cpm.loc[gene, col]})
tidy = pd.DataFrame(records)
age_order = sorted(tidy['age_mo'].unique())

# --- Fig 1: expression over age ---
fig, axes = plt.subplots(1, len(GENES), figsize=(5 * len(GENES), 4))
for ax, gene in zip(axes, GENES):
    data = tidy[tidy['gene'] == gene]
    sns.boxplot(
        data=data, x='age_mo', y='expr', order=age_order,
        hue='sex', hue_order=['Female', 'Male'], palette=SEX_COLORS,
        linecolor='black', linewidth=1, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=data, x='age_mo', y='expr', order=age_order,
        hue='sex', hue_order=['Female', 'Male'],
        size=4, jitter=True, dodge=True, palette='dark:black', legend=False, ax=ax,
    )
    ax.set_xlabel('Age (months)')
    ax.set_ylabel('log2(1+CPM)')
    ax.set_title(f'{gene} expression over age')
    ax.legend(title='Sex', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_FIG}')
