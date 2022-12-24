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


def training(train_dataloader, valid_dataloader):
    '''
        main function of train
    '''
    ## wanb to record process
    wandb.init(project="img-prg-cap", entity="forbes-dl")

    model = ImgCapModel().to(device)

    optimizer_choices = {
        'AdamW': torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        ),
        'Adafactor': Adafactor(
            model.parameters(),
            lr=1e-4,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    }
    optimizer = optimizer_choices[args.optimizer]

    ## scheduler check
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accu_step)
    max_train_steps = args.num_epoches * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.grad_accu_step,
        num_training_steps=max_train_steps * args.grad_accu_step,
    )

    # do f1 for these two?
    metric = evaluate.load("rouge")
    metric = evaluate.load("bleu")

    ## check config and save to wandb
    # config = wandb.config
    wandb.config.update(args) 
    for arg in vars(args):
        print(f"<{arg}>: {getattr(args, arg)}")
        print("==================================")


    print("***Start training***")


    for epoch in range(args.num_epoches):
        print(f"-----Epoch[{epoch+1}/{args.num_epoches}]-----")
        model.train()
        train_loss = 0

        # per batch
        train_loss = train_per_epoch(
            model,
            optimizer_choices[args.optimizer],
            lr_scheduler,
            train_dataloader,
            epoch,
            args
        )

        wandb.log({"train_loss": train_loss})

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

        pr_list = {
            "preds": predictions,
            "refs": references
        }

        # rouge/bleu score
        # result = metric.compute(references=references, predictions=predictions)
        print(scores)
        wandb.log(scores)


        # save preds of validation set to check each epoch
        save_preds(epoch, pr_list)

        # save and check
        model_path = os.path.join("test/", "model_{}".format(epoch + 1))

        model.save_model(model_path)
        print('Train Loss: {:3.6f} | Val Loss: {:3.6f}'.format(train_loss, 0,))

    ## save the best model to wandb
    wandb.save(model_path)



def main():
    " load data "
    raw_datasets = load_raw_datasets(args)

    " dataset and dataloader"
    train_dataset = ImgCapDataset(raw_datasets["train"])
    print('train dataset format:')
    print(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = ImgCapDataset(raw_datasets["valid"])
    print('valid dataset format:')
    print(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_dataset.collate_fn
    )

    "train/valid"
    training(train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()
