'''
    for params setting
'''

from argparse import ArgumentParser, Namespace
from typing import *
import torch
from transformers import SchedulerType


def parse_args() -> Namespace:
    parser = ArgumentParser()

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

    '''output files'''
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
