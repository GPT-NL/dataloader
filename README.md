<div style="text-align: center;">

  <h1>
    <a href="https://gpt-nl.nl/" target="_blank"><img src="https://gpt-nl.nl/publish/pages/5180/beeldmerk-gpt_nl.svg" alt="GPT-NL" width="200"></a>
    <br>
    Data loader
  </h1>

#### Bucketing, tokenizing, and distributing mixing of structured data, supporting the construction of the <a href="https://gpt-nl.nl" target="_blank">GPT-NL</a> model.

<a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
<a href="#usage">Usage</a> &nbsp;&middot;&nbsp;
<a href="#development-notes">Development notes</a>

</div>

> This repository is part of [GPT-NL](https://gpt-nl.nl/), a responsible alternative to existing language models, using lawfully obtained Dutch data and sharing proceeds with copyright holders to promote fairness and strengthen Europe's digital position.
> We aim to be as transparent as possible about how the model has been created. This open source repository should be considered as an effort to support this transparency and fairness, not to be seen as a working product of any kind.  
> For any questions regarding GPT-NL, check our [website](https://gpt-nl.nl/). Any questions regarding this specific repository can be submitted as an issue in the issue tracker of this repository.

This repository contains code for transforming curated text data into tokenized datasets ready for GPT-NL pre-training.
It implements a full pipeline from fitting the tokenizer to categorising the data (e.g. by quality or language) to producing tokenized data mixtures that leverage these categories.
The code is written in Python and designed to run efficiently on a SLURM-managed HPC cluster (Snellius managed by Surf in our case).
Note that it is tailored specifically to GPT-NL and will require adaptation for other use cases.

## Installation

> :information_source: If you do not have [Poetry](https://python-poetry.org/) installed, do that first. Windows users are recommended to use WSL.

If using a high-performance cluster, load an appropriate module:
```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0  # Load a valid python version (>=3.10,<4.0)
which python # See the python path
poetry env use THIS_PYTHON_PATH
poetry install
```

Otherwise, make sure a python version >=3.10 and <=3.12 is used.

```bash
which python # See the python path
poetry env use THIS_PYTHON_PATH
poetry install
```

<!--
Consider:

- You can assume a basic level of understanding. Not _every_ step has to be explained extensively - often listing the commands is sufficient. Example: you don't have to explain the concept of virtual environments.
- Be complete and assume that people are starting fresh. Example: don't assume people have a specific conda or a specific Python version.
- People use different operating systems.
-->

## Usage

![Diagram of the five stages of the data loader flow: 1. Curated Output, 2. Bucketing, 3. Tokenizing, 4. Distribution mixing, and 5. Final file pattern.](./assets/dataloading-GPT-NL.png)
 
Getting this output ready for training (dataloading) consists of the following steps (visualised in the image above):

### A. Investigate & assemble meta data for output curation (part of Step 2 bucketing)

This is done in the notebook `notebooks/explore-curated-data.ipynb` and uses metadata provided by Work Package 18, see [section 2.2.2 of the System Architecture document: Training Pipeline - GPTNL-DEL-4002](https://resolver.tno.nl/uuid:2e3fdebe-7284-4539-a863-0b7f1c3bcc3b) and `dataset_metadata/datasets_example.yaml`.
As an output we save an overview of currently curated datasets with the metadata in `curated_data_overview.md` and in `curated_data_overview.csv`.

### B. Tag the curated data with assembled metadata (part of Step 2 bucketing)

> NB. Before starting these steps, you can run `gptnl_dataloader/bucketing/copy_curated_sample.sh` for a clean symlinked directory, so you do not have to work directly in the official curation output folder.

This stage is configured through `gptnl_dataloader/bucketing/config.env` and uses dataset-level metadata from the YAML file referenced by `QUALITY_METADATA_FILE`.

The tagging step is implemented in `gptnl_dataloader/bucketing/tag.py` and can be submitted with `gptnl_dataloader/bucketing/tag_slurm_script.sh`.

The code adds deterministic metadata columns to each parquet row:

1. `doc_id`
2. `dataset_quality`
3. `is_code`
4. `is_new_epoch_2`
5. `original_file_path`

Quality, code/non-code, and epoch flags come from the metadata YAML. The parquet files keep their original names, row counts, and per-dataset directory structure. The stage also tracks progress in `processing_status.json` and writes a `document_ids_registry.json` for downstream validation.

### C. Bucketing & merging (part of Step 2 bucketing)

This stage is implemented in `gptnl_dataloader/bucketing/bucketing.py` and can be submitted with `gptnl_dataloader/bucketing/bucketing_slurm_script.sh`.

The bucketing logic uses `is_code`, `is_new_epoch_2`, `language`, and `dataset_quality` from the tagged parquet files. The current pipeline produces eight buckets:

1. `new_high_quality`
2. `new_low_medium_quality`
3. `dutch_high_quality`
4. `dutch_low_medium_quality`
5. `english_high_quality`
6. `english_low_medium_quality`
7. `other_languages`
8. `code`

Processing happens in batches and in parallel. Rows are first written to temporary bucket files and then merged into more uniform output parquet files. Merging can be driven by row count or by target file size; the current Slurm script uses file size.

This stage does not use a separate perplexity-analysis step. There is no active `2_ppl_analysis.py` in this directory, and bucket assignment is not based on PPL thresholds.

### D. Post-bucketing validation

Validation is implemented in `gptnl_dataloader/bucketing/post_bucketing_check.py`.

It uses the registry from the tagging step plus the bucket outputs to check:

1. missing document IDs
2. duplicate IDs within a bucket
3. duplicate IDs across buckets
4. dataset-level coverage mismatches

### E. Train/validate split

Keeps a disjoint set of data for validation (2%)

```bash
poetry run python -m gptnl_dataloader.train_validation_split r
```

### F. Fit the tokenizer (part of Step 3 tokenizing)

Train a tokenizer on a representative subset of the data.

To train the tokenizer from scratch, there are a couple of steps
1. Prepare the training data in the right format for the [`sentencepiece`](https://pypi.org/project/sentencepiece/) library. Additionally, this step includes a sampling strategy to extract a subset of the data for the actual tokenization training task. The sampling is done on file level and the percentages represent the number of files from the original source to be included in the subset of that source. So 90% of Dutch would not mean that the tokenization set includes 90% Dutch data, but rather that 90% of the original Dutch data is sampled.

This can be done by
```bash
poetry run python -m gptnl_dataloader.tokenization.parquet_to_txt --base_dir <data_path> --output_txt <output.txt> 
# Or on SLURM in parallel
sbatch parquet_to_txt.job
```
2. Train the tokenizer using the provided  `sentencepiece` config at `gptnl_dataloader/tokenization/config/gpt-nl.yaml`
```bash
poetry run python -m gptnl_dataloader.tokenization.fit_tokenizer --output_path <output_path> --config_path <config/.yaml>
# Or on SLURM in parallel 
sbatch fit_tokenizer.job
```
This will yield a sentencepiece .model and the converted tokenizer in HuggingFace format at `output_path/hf`

## G. Tokenize the curated data ([src code](./gptnl_dataloader/tokenization/))
Given a HuggingFace tokenizer, tokenize the full dataset in parallel
```bash
poetry run python tokenize_text.py \
        --data_path <data_path> \
        --output_dir <output_dir> \
        --cache_dir <cache_dir> \
        --remove_columns <column_names> \
        --tokenizer <tokenizer_path> \
        --data_type pretraining \
        --file_pattern <file_pattern> \
        --eos <eos_token_id> \
        --num-proc <_nprocs> \
```
Or run it on SLURM in parallel which will prepare the tokenize script
```bash
# Prepare the tokenize array job 
./setup_tokenize_array.job
# Then actually run tokenization
sbatch tokenize_job_array.job
```

### H. Create the data mixtures for the two phases of training (part of Step 4 distribution mixing)

Re-split and restore the bucket structure for the tokenized files (‼️only applies to v1 of the dataset. With v2 the tokenization already preserves subdirectories in `buckets_split_train` and `buckets_split_validate`, so this step is not required). For v1, the tokenized files are stored in a single directory without bucket folders, and you need to run this command: `poetry run python gptnl_dataloader/train_validation_split/resplit_files_tokenized.py`

Then run the sampling with `poetry run python -m gptnl_dataloader.sampling`.


Organize the tokenized data into sets ready for training

- Main training (trapezoid primary/constant LR phase) ~80% of the data
- Final stage (cooldown end of trapezoid learning/annealing) %15% of the data <-- higher quality data

## Development notes

In the `gptnl_dataloader/inspecting` you can find a useful script to search for a string in parquet files
