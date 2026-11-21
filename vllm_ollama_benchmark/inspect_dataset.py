from mmengine.config import read_base
from copy import deepcopy

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets


# ---------------------------
# Sample configuration helper
# ---------------------------

def limit_samples(dataset_configs, sample_count):
    dataset_configs = deepcopy(dataset_configs)

    for cfg in dataset_configs:
        reader_cfg = cfg.get('reader_cfg', {})

        # Apply test sample limit
        reader_cfg['test_range'] = sample_count

        cfg['reader_cfg'] = reader_cfg

    return dataset_configs


# ---------------------------
# Apply sampling
# ---------------------------

gsm8k_sampled = limit_samples(gsm8k_datasets, 100)
mmlu_sampled = limit_samples(mmlu_datasets, 100)
ifeval_sampled = limit_samples(ifeval_datasets, 50)

# Combine datasets
datasets = gsm8k_sampled + mmlu_sampled + ifeval_sampled


# datasets = gsm8k_datasets + mmlu_datasets + ifeval_datasets

for cfg in datasets:
    print("=" * 80)
    print("ABBR:", cfg.get("abbr"))
    print("CONFIG KEYS:", cfg.keys())
    print("READER CONFIG:", cfg.get("reader_cfg"))
    print("DATASET CONFIG:", cfg.get("dataset"))