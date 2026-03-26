# CLAUDE.md

## Repository Structure

```
SingleCellArchetype/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── data/                  # sample data (tracked)
├── local_data/            # local/large data (gitignored)
├── docs/                  # documentation
├── scripts/               # standalone scripts
└── src/
    └── SingleCellArchetype/
        ├── __init__.py
        ├── main.py
        ├── utils.py
        ├── tutorial_complete.ipynb
        └── tutorial_minimum.ipynb
```

## Git Branches

- `main`: stable release branch
- `dev`: development branch
- `dev2`: active development branch (branched from `dev`)

## Git Configuration

- user.name: FangmingXie
- user.email: fmxie1993@gmail.com

## Environment

- Conda env to use to run this project: `archetype`  
- Key dependencies: `anndata`, `py_pcha`

## .gitignore Notes

- `local_data/` is gitignored (for large or local-only data files)

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
