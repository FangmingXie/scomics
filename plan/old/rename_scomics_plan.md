# Plan: Complete SingleCellArchetype → scomics Rename

## Context

The project was renamed from `SingleCellArchetype` (on branch `dev2`) to `scomics` (on `main`). The git remote already points to `https://github.com/FangmingXie/scomics.git`. However, many files were not updated during the rename — the `src/` package directory still carries the old name, imports are broken, pyproject.toml has the wrong name, and several scripts have hardcoded paths to the old filesystem location.

---

## Changes Required

### 1. `src/SingleCellArchetype/` → `src/scomics/` (structural)
Use `git mv src/SingleCellArchetype src/scomics` to rename the package directory. This fixes all Python imports once the package is reinstalled.
- Also rename `src/SingleCellArchetype.egg-info/` → rebuild via `pip install -e .` (no manual rename needed; it's gitignored)
- Update docstrings in `src/scomics/main.py` line 1 and `src/scomics/utils.py` line 1

### 2. `pyproject.toml`
- Line 2: `name = "SingleCellArchetype"` → `name = "scomics"`

### 3. `CLAUDE.md`
- Line 6: `SingleCellArchetype/` → `scomics/`
- Line 20: `src/SingleCellArchetype/` → `src/scomics/`
- Line 30: Remove `dev2` branch entry (it's now the `main` branch)

### 4. Python imports in scripts (~10 files)
All `from SingleCellArchetype` → `from scomics` in:
- `scripts/common.py`
- `scripts/cheng22/00.tutorial_complete.py`
- `scripts/cheng22/00.tutorial_minimum.py`
- `scripts/cheng22/01.tutorial_per_sample.py`
- `scripts/cheng22/02.num_archetype_3d.py`
- `scripts/cheng22/02.num_archetype_3d_ndim3.py`
- `scripts/cheng22/02.num_archetypes.py`
- `scripts/cheng22/03.num_archetype_rep.py`
- `scripts/cheng22/04.num_archetype_cheng22.py`
- `scripts/dombrovski25/04.num_archetype_dombrovski25.py`

### 5. Hardcoded filesystem paths (3 files)
`/home/qlyu/mydata/project/SingleCellArchetype/` → `/home/qlyu/mydata/project/scomics/` in:
- `scripts/morcom26/16.label_transfer_cheng22yoo25_to_morcom26.py` (line 19 `REF_FILE`)
- `scripts/morcom26/17.visualize_label_transfer.py` (line 16 `REF_FILE`)
- `scripts/morcom26/18.visualize_label_transfer_ENL23CTX.py` (line 16 `REF_FILE`)

(Confirmed: `/home/qlyu/mydata/project/scomics/` exists with `local_data/source/` subdirectory)

### 6. `README.md` (optional cosmetic)
Multiple references to `FangmingXie/SingleCellArchetype` in badge URLs, image URLs, and git clone commands. Low priority since the README isn't actively maintained.

### 7. Docs (low priority)
- `docs/plan_per_sample_analysis.md` — references to `src/SingleCellArchetype/`
- `docs/04.num_archetype_dombrovski25_plan.md` — references to `SingleCellArchetype.main.SCA`

---

## Execution Order

1. `git mv src/SingleCellArchetype src/scomics`
2. Edit `pyproject.toml`
3. Edit `CLAUDE.md`
4. Edit `src/scomics/main.py` and `src/scomics/utils.py` docstrings
5. Edit all 10 script files (imports)
6. Edit 3 morcom26 scripts (hardcoded paths)
7. Run `pip install -e .` in the `archetype` conda env to rebuild the package with the new name
8. Optionally update `README.md` and docs

---

## Verification

After changes:
```bash
conda run -n archetype python -c "import scomics; print('OK')"
conda run -n archetype python scripts/common.py  # should not ImportError
```
Also confirm the morcom26 REF_FILE paths resolve:
```bash
ls /home/qlyu/mydata/project/scomics/local_data/source/cheng22_yoo25/
```
