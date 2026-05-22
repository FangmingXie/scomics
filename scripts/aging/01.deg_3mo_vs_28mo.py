import os
import numpy as np
import pandas as pd
from scipy import stats

# --- file paths ---
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'links', 'aging', 'hahn23_vis_bulk.parquet')
RES_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'res', 'aging')
OUT_DEG = os.path.join(RES_DIR, '01.deg_3mo_vs_28mo_male.csv')

# --- config ---
AGE_A = '3mo'
AGE_B = '28mo'
SEX = 'Male'

# === main ===
os.makedirs(RES_DIR, exist_ok=True)

# --- load and normalize ---
df = pd.read_parquet(INPUT_FILE)
cpm = df.div(df.sum(axis=0), axis=1) * 1e6
log2cpm = np.log2(1 + cpm)

# --- select samples (male only to avoid sex confound) ---
cols_a = [c for c in df.columns if c.startswith(f'{AGE_A}_{SEX}')]
cols_b = [c for c in df.columns if c.startswith(f'{AGE_B}_{SEX}')]
print(f'{AGE_A} {SEX}: {cols_a}')
print(f'{AGE_B} {SEX}: {cols_b}')

ga = log2cpm[cols_a].values
gb = log2cpm[cols_b].values

# --- DEG: t-test + log2FC ---
lfc  = gb.mean(axis=1) - ga.mean(axis=1)
pval = np.array([stats.ttest_ind(gb[i], ga[i]).pvalue for i in range(len(lfc))])

res = pd.DataFrame({'gene': log2cpm.index, 'lfc': lfc, 'pval': pval})
res = res.sort_values('pval').reset_index(drop=True)
res.to_csv(OUT_DEG, index=False)
print(f'Saved {OUT_DEG}')

# --- report top hits ---
print(f'\nTop 20 upregulated ({AGE_B} vs {AGE_A}, {SEX}):')
print(res[res['lfc'] > 0].head(20).to_string(index=False))

print(f'\nTop 20 downregulated ({AGE_B} vs {AGE_A}, {SEX}):')
print(res[res['lfc'] < 0].head(20).to_string(index=False))
