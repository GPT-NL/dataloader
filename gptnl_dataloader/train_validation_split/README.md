# Train and validation split

This stage separates the data into train (98%) and validation (2%).

The split is:
- stratified across datasets
    - dataset comes from parquet `original_file_path`: `gpt_nl_dataset_vX.X/DATASET` (column `dataset_name` has variations, e.g. `Instituut voor de Nederlandse Taal - DAESO 1.0`)
    - in this way validation set has samples from all the datasets
- validation percentage is kept quite low: `2%` to let most of the data for training and at the same time reduce validation time

## How to run

```bash
# (needs more memory, otherwise is killed: salloc -p rome --mem=120GB  -t 05:00:00)
# inspect the files
python -m gptnl_dataloader.train_validation_split i
# do the split
python -m gptnl_dataloader.train_validation_split r
# verify the stats after split
```

## Dividing again files tokenized (need folder structure in buckets for sampling)

‼️only applies to v1 of the dataset. With v2 the tokenization already preserves subdirectories in `buckets_split_train` and `buckets_split_validate`, so this step is not required.

- Problem: `GPTNL_1_TOKENIZED/buckets_split_train/` and `GPTNL_1_TOKENIZED/buckets_split_validate/` are flat directories
- Solution: run `python gptnl_dataloader/train_validation_split/resplit_files_tokenized.py`
- Outputs: `GPTNL_1_TOKENIZED/splitted_buckets_train/` and `GPTNL_1_TOKENIZED/splitted_buckets_validate`

## compare rows

This code can be used to check that the train/validation split worked properly.

```python
import pyarrow.parquet as pq
import os
from glob import glob
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

p_base = os.environ["GPTNL_1_BUCKETED"]
buckets = [
    "english_low_medium_quality",
    "english_high_quality",
    "dutch_low_medium_quality",
    "dutch_high_quality",
    "code",
    "other_languages",
]
tot = 0
for bucket in buckets:
    p1 = "bucketing_output"
    p2 = "buckets_split_train"
    p3 = "buckets_split_validate"
    file_names = glob(f"{p_base}/{p1}/{bucket}/*.parquet")
    tot += len(file_names)
    file_name_bases = [file_name.split("/")[-1].replace(".parquet", "") for file_name in file_names]
    # for file_name_base in tqdm(file_name_bases): # takes too long
    # verify all the files are there
    l2 = glob(f"{p_base}/{p2}/{bucket}/*.parquet")
    l3 = glob(f"{p_base}/{p3}/{bucket}/*.parquet")
    assert len(file_names) == len(l2) == len(l3), f"{bucket} {len(file_names)} {len(l2)} {len(l3)}"
    # verify that the first file also has matching number of rows
    assert file_name_bases, f"{bucket}"
    file_name_base = file_name_bases[0]
    fp1 = f"{p_base}/{p1}/{bucket}/{file_name_base}.parquet"
    fp2 = f"{p_base}/{p2}/{bucket}/{file_name_base}_train.parquet"
    fp3 = f"{p_base}/{p3}/{bucket}/{file_name_base}_validate.parquet"
    s1 = os.path.getsize(fp1) >> 20 # 903 MB
    s2 = os.path.getsize(fp2) >> 20 # 806 MB
    s3 = os.path.getsize(fp3) >> 20 # 16 MB
    print("size", s1, s2, s3, (s2 + s3)/s1) # ratio is 0.91
    l1 = len(pq.read_table(fp1)) # 116k
    l2 = len(pq.read_table(fp2)) # 114k
    l3 = len(pq.read_table(fp3)) # 114k
    ratio = (l2 + l3)/l1
    print("rows", l1, l2, l3, ratio) # ratio is 1% --> no rows lost
    assert (ratio == 1)

print(tot)
```
