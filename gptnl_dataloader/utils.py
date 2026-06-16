import sys
import os
from dotenv import load_dotenv

load_dotenv()

# GPTNL_SCRATCH/gpt_nl_dataset_v2.0_bucketed/bucketing_output
DATASET_DEFAULT_PATH = os.path.join(
    os.environ["GPTNL_SCRATCH"], "gpt_nl_dataset_v2.0_bucketed"
)
# DATASET_DEFAULT_PATH = os.environ["GPTNL_2_BUCKETED"]
# TOKENIZED_DATASET_ROOT_PATH = os.environ["GPTNL_2_TOKENIZED"]
TOKENIZED_DATASET_ROOT_PATH = os.environ["GPTNL_2_TOKENIZED"]

# located under DATASET_DEFAULT_PATH
BUCKETS_OUTPUTS_DEFAULT_NAME = "bucketing_output"

# located under DATASET_DEFAULT_PATH
BUCKETS_TRAIN_SPLIT_DEFAULT_NAME = "buckets_split_train"
# located under DATASET_DEFAULT_PATH
BUCKETS_VALIDATE_SPLIT_DEFAULT_NAME = "buckets_split_validate"

ALL_BUCKETS_DEFAULT = [
    "new_high_quality",
    "new_low_medium_quality",
    "english_high_quality",
    "english_low_medium_quality",
    "dutch_high_quality",
    "dutch_low_medium_quality",
    "code",
    "other_languages",
]


# automatically redirect all prints to file
# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
