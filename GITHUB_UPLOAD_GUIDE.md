# GitHub Release Notes

This file explains how the current public Ultra-LSNT repository is organized on GitHub.

## Current release intent

The repository is maintained as a paper-aligned public release for:

- code inspection
- reproducibility checking
- access to released data assets and split manifests
- access to reported figures and result tables

## Important layout choices

### Canonical data location

The canonical location for released data assets is:

- `data/raw/`
- `data/processed/`

### Compatibility copies

Some root-level CSV files are intentionally retained because older scripts in the repository expect filenames such as `wind_final.csv` or `wind_us.csv` in the repository root. Those files are compatibility copies, not a separate storage layout.

### Documentation split

- Root-level markdown files are the public-facing documentation.
- `docs/` keeps archived reports and supplementary notes.

## Recommended reader flow

1. Read `README.md`
2. Follow `QUICKSTART.md`
3. Check `DATA.md`
4. Use `EXPERIMENTS.md` for the main scripts
5. Use `STRUCTURE.md` to navigate the repository

## Maintainer note

When updating the public release, keep the following in sync:

- manuscript title references in `README.md`
- released data asset claims in `DATA.md`
- script entry points in `QUICKSTART.md` and `EXPERIMENTS.md`
- repository map in `STRUCTURE.md`
