from mmengine.config import read_base
from opencompass.models import OpenAISDK

# Import dataset configuration
with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    # from opencompass.datasets.mmlu import mmlu_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_dataests
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
datasets = gsm8k_datasets

# Model configuration
models = [
    dict(
        type=OpenAISDK,
        abbr='deepseek-r1-latest-ollama',
        path='deepseek-r1:latest',

        openai_api_base='http://localhost:11434/v1',
        key='ollama',

        max_seq_len=8192,
        max_out_len=1024,

        temperature=0,
        batch_size=1,

        run_cfg=dict(num_gpus=0),
    )
]


# from mmengine import read_base


# with read_base():
#     from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

# datasets = gsm8k_datasets


# # with read_base():
# #     from opencompass.datasets import gsm8k

# # datasets = gsm8k