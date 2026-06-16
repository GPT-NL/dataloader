# Sampling / data mixtures

This implements file-based sampling.

We need to have a certain distribution across language and quality.

Inputs:
- buckets of data (at the moment parquet files, but will be tokenized files)
- desired proportion
    - how much data to keep for validation
    - how much of each language for data mixture

Outputs:
- list of paths for validation
- list of paths to use for the different training data mixtures (primary and annealing)

## Running this script

```bash
poetry run python -m gptnl_dataloader.sampling
# v1
poetry run python -m gptnl_dataloader.sampling --dataset-root=GPTNL_1_TOKENIZED --file-name-pattern="*.npy"
# v2
poetry run python -m gptnl_dataloader.sampling --dataset-root=GPTNL_2_TOKENIZED --file-name-pattern="*.npy"
```
