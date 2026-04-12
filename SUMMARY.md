# Project Summary

Ultra-LSNT is a public reproducibility repository for the manuscript:

> "Controller-side feasibility guidance for wind power forecasting under corrupted SCADA"

## What the repository provides

- the Ultra-LSNT codebase
- baseline comparison scripts
- released Wind (CN) raw and processed assets needed for the public workflow
- processed benchmark datasets for secondary diagnostics
- split manifests used by the reported chronological evaluation protocol
- manuscript-supporting figures and tables

## What the repository is optimized for

This repository is optimized for:

- paper-aligned reproducibility
- experiment inspection
- script-based reruns of the main workflows

It is not primarily optimized as a packaged machine-learning library.

## Recommended starting points

- `README.md` for the high-level overview
- `QUICKSTART.md` for a first local run
- `DATA.md` for the released data assets
- `EXPERIMENTS.md` for the main scripts
- `STRUCTURE.md` for the repository map
- `docs/README.md` for archived material and maintainer notes

## Practical notes

- The canonical data location is `data/`.
- Root-level CSV copies are retained only for backward compatibility with older scripts.
- The repository includes both code and released artifacts so the paper claims can be checked against concrete files.
