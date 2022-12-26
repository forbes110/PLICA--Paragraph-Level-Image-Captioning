'''
    for params setting
'''

from argparse import ArgumentParser, Namespace
from typing import *
import torch
from transformers import SchedulerType


def parse_args() -> Namespace:
    parser = ArgumentParser()

    ''' raw files '''
    parser.add_argument(
        "--raw_train_split", default='./data/raw/train_split.json', type=str
    )
    parser.add_argument(
        "--raw_valid_split", default='./data/raw/valid_split.json', type=str
    )
    parser.add_argument(
        "--raw_test_split", default='./data/raw/test_split.json', type=str
    )

    ''' processed files'''
    parser.add_argument(
        "--dataset_name", default=None, type=str
    )
    parser.add_argument(
        "--train_file", default='./data/processed/paragraphs/train_file.json', type=str
    )
    parser.add_argument(
        "--valid_file", default='./data/processed/paragraphs/valid_file.json', type=str
    )
    parser.add_argument(
        "--test_file", default='./data/processed/paragraphs/test_file.json', type=str
    )

    ''' training configs '''
    parser.add_argument(
        "--few_data_test", action="store_true", help="Sampling 100 data for testing correctness."
    )
    parser.add_argument(
        "--use_pretrain_imgcap", action="store_true", help="Using pretrained img caption model."
    )
    parser.add_argument(
        "--batch_size", default=4, type=int
    )
    parser.add_argument(
        "--num_epoches", default=10, type=int
    )
    parser.add_argument(
        "--grad_accu_step", default=8, type=int
    )

    parser.add_argument(
        "--use_L1reg", default=False, type=bool
    )
    parser.add_argument(
        "--lambda_val", default=1e-5, type=float
    )
    parser.add_argument(
        "--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=torch.device
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default='linear',
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0)
    parser.add_argument(
        "--optimizer",
        type=str,
        default='AdamW',
        help="The optimizer type to use.",
        choices=["AdamW", "Adafactor"]
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    '''predict configs'''
    parser.add_argument(
        "--model_path", default=None, type=str
    )
    parser.add_argument(
        "--repetition_penalty", default=1.0, type=float
    )
    parser.add_argument(
        "--no_repeat_ngram_size", default=0, type=int
    )
    parser.add_argument(
        "--val_max_target_length", default=60, type=int
    )
    parser.add_argument(
        "--num_beams", default=1, type=int
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Sample or not"
    )

    args = parser.parse_args()
    return args
