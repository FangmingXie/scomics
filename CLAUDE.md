# CLAUDE.md

## Repository Structure

```
scomics/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── data/                  # sample data (tracked)
├── local_data/            # local/large data (gitignored)
│   ├── res/                  # results (computed data - csv, parquet, etc)
│   └── fig/                  # visualization (png, html, etc)
├── links/                 # curated symlinks to data inputs, organized by sub-project (gitignored)
│   ├── astro/                # astrocyte sub-project (e.g. gao25_scrna_astro.h5ad)
│   ├── it/                   # IT neuron sub-project (e.g. cheng22, yoo25, gao25 IT datasets)
│   └── l23_evo/              # L2/3 evolution sub-project (e.g. jorstad23 human L23 IT)
├── plan/                  # plans organized by project
│   ├── astro/             
│   ├── ...                
│   └── l23_evo/           
├── docs/                 # documentation
├── scripts/               # standalone analysis scripts, organized by projects 
│   ├── common.py          # shared utilities for scripts
│   ├── cheng22/           # Cheng et al. 2022 analysis
│   ├── dombrovski25/      # Dombrovski et al. 2025 analysis
│   └── morcom26/          # Morcom et al. 2026 analysis
└── src/
    └── scomics/
        ├── __init__.py
        ├── main.py
        └── utils.py
```

## Git Branches

- `main`: active development branch

## Git Configuration

- user.name: FangmingXie
- user.email: fmxie1993@gmail.com

## Environment

- Use this conda env to run this project: `archetype`, that means to run any python script with `conda run -n archetype`
- For heavy-lifting scripts, prefer running unbuffered at both levels so output (progress, logs) streams in real time: use `conda run --no-capture-output -n archetype python -u <script>.py`

## .gitignore Notes

- `local_data/` is gitignored (for large or local-only data files)
- `links/` is tracked (curated symlinks to data inputs)

## coding styles
- Define all file paths (input and output files) in the beginning of each script as much as possible. Capitalize the variables that store these file paths.

**Simplify Relentlessly**: Remove complexity aggressively - the simplest design that works is usually best

#### Fail-Fast, No Fallbacks
- **No Silent Fallbacks**: Code must fail immediately when expected conditions aren't met. Silent fallback behavior masks bugs and creates unpredictable systems.
- **Explicit Error Messages**: When something goes wrong, stop execution with clear error messages explaining what failed and what was expected.
- **Example**: `raise ValueError(f"Required model {model_name} not found")` instead of falling back to first available model.

### ⚠️ **IMPORTANT: Rewrite Project - Breaking Changes Encouraged**

**This package is a complete rewrite**, not an actively used codebase with external dependencies. This means:

- **Breaking changes are encouraged** when they follow best practices
- **No backward compatibility constraints** - optimize for clean architecture
- **Clean module organization** - each module has a single, clear purpose

This approach ensures the codebase remains maintainable and forces explicit dependencies that make the architecture clear to all developers.

## Claude Code Automation Rules
- When operating in Plan Mode, ALWAYS save the finalized implementation plan as a distinct markdown file under the `plan/` folder before concluding the turn.
- Never execute modifications while Plan Mode is toggled active.

## ⚠️ **IMPORTANT: Installing a package
- do not attempt to install new packages without asking permission explicitly. 
- before installing anything new, save a copy of the current package list under `env/`. 
- always try using conda first, and pip later. 