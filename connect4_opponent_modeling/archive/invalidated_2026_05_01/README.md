# Invalidated On 2026-05-01

This archive contains code and artifacts that were removed from the live
experiment surface during the May 1, 2026 defensibility cleanup.

## Why These Items Were Archived

- benchmark paths that did not faithfully implement the benchmark they claimed
- mixed-protocol training code centered on deprecated SFT-first workflows
- narrative markdown summaries that were stronger than the underlying canonical evidence
- smoke checkpoints and exploratory logs that were cluttering the active repo surface

## Archive Structure

- `eval/`: invalid or non-canonical benchmark wrappers
- `training/`: deprecated training entrypoints and SFT-only workflow files
- `results/`: superseded narrative result summaries
- `backups/`: exploratory logs, smoke checkpoints, and old backup directories

These files are preserved for project memory and debugging only. They should
not be used as the basis for new experiment claims.
