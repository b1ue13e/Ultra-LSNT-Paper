# Contributing

Thanks for your interest in improving Ultra-LSNT.

This repository is maintained as a paper-aligned public release, so the most helpful contributions are usually:

- bug reports tied to a concrete script or file
- reproducibility fixes
- documentation improvements
- path, environment, or compatibility fixes that make the public release easier to run

## Before opening a change

Please try to include:

- the exact script or file involved
- the command you ran
- the operating system and Python version
- the observed error message or incorrect behavior
- the expected behavior

## Pull request guidance

Small, focused pull requests are preferred.

Please avoid mixing:

- documentation cleanup
- experiment logic changes
- data-layout changes

in a single PR unless they are tightly connected.

## Data and reproducibility notes

- Canonical data files live under `data/`
- Root-level CSV copies are retained for backward compatibility with older scripts
- If you change a path expectation, please update the relevant documentation files too:
  - `README.md`
  - `QUICKSTART.md`
  - `DATA.md`
  - `STRUCTURE.md`

## Code style

- Prefer minimal, targeted changes
- Keep backward compatibility when practical
- Avoid introducing extra dependencies unless clearly necessary
- Update docs when user-facing behavior changes

## Citation-sensitive repository

Because this repository is tied to a submitted manuscript, please be careful with changes that would materially alter:

- the public data-availability claim
- the stated experiment entry points
- the interpretation of released results

For those cases, documentation updates should accompany code changes.
