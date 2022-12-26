import os
import time
import evaluate
from transformers import (
    Adafactor,
    get_scheduler
)

from torch.utils.data import DataLoader
from config import parse_args
from tqdm.auto import tqdm
import torch

from model.ImgCapModel import ImgCapModel
from utils.ImgCapDataset import ImgCapDataset
from utils.train_utils import (
    save_preds,
    train_per_epoch,
    valid_per_epoch,
    load_raw_datasets
)
import math
import wandb

'''
    configs
'''
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def testing(valid_dataloader):
    '''
        main function of train
    '''
    ## wanb to record process
    wandb.init(project="img-prg-cap", entity="forbes-dl")

    model = ImgCapModel(args.use_pretrain_imgcap).to(device)

    model.load_model(args.model_path)

    wandb.config.update(args) 
    for arg in vars(args):
        print(f"<{arg}>: {getattr(args, arg)}")
        print("==================================")


    print("***Start predicting***")
    model.eval()
    gen_kwargs = {
        "max_length": args.val_max_target_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        # "top_k": args.top_k,
        # "top_p": args.top_p,
        # "temperature": args.temperature,
        # "repetition_penalty": args.repetition_penalty,
        # "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    # output: val_ids(for check image), predictions, reference and scores from metrics
    val_ids, predictions, references, scores = valid_per_epoch(
        model,
        valid_dataloader,
        gen_kwargs
    )

    print(scores)
    wandb.log(scores)



def main():
    " load data "
    raw_datasets = load_raw_datasets(args)

    valid_dataset = ImgCapDataset(raw_datasets["valid"])
    print('valid dataset format:')
    print(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_dataset.collate_fn
    )

    "train/valid"
    testing(valid_dataloader)


if __name__ == '__main__':
    main()
