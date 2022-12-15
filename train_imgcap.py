import os
import time
import numpy as np
from datasets import load_dataset
from transformers import Adafactor

import torch
from torch.utils.data import DataLoader, Dataset
from config import parse_args
from tqdm.auto import tqdm
import evaluate

from models.ImgCapModel import ImgCapModel
from utils.ImgCapDataset import ImgCapDataset

import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


'''
    configs
'''
args = parse_args()


def train_per_epoch(model, optimizer, grad_accum_epoches, train_dataloader):
    train_loss = 0

    train_bar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=True, ncols=100)

    # for idx, (images, captions) in enumerate(train_loader):
    for idx, (images, captions) in enumerate(train_bar):
        print("\r", idx, end="")
        outputs = model(images, captions)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        if ((idx+1) % grad_accum_epoches == 0):
            optimizer.step()
            optimizer.zero_grad()

    train_loss = train_loss / len(train_per_epoch)
    return train_loss


def valid_per_epoch(model, optimizer, batch_size, valid_dataloader, metric):

    ## to save result pair
    predictions, references = [], []
    valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader), position=0, leave=True, ncols=100)
    for idx, (images, captions) in enumerate(valid_bar):
        gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample, 
            "top_k": args.top_k, 
            "top_p": args.top_p,
            "typical_p": args.typical_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        }
        with torch.no_grad():
            # ...

            decoded_preds, decoded_labels = 0, 0
            predictions, references = 0, 0
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )

        return predictions, references, metric

def training(train_dataloader, valid_dataloader):
    iteration = args.num_epoches
    grad_accum_epoches = 8
    model = ImgCapModel()

    optimizer = Adafactor(
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

    # do f1 for these two?
    metric = evaluate.load("rouge")
    metric = evaluate.load("bleu")

    ## to save results
    print("***Start training***")

    for epoch in range(iteration):
        print(f"\n=== Epoch {epoch+1} ===")

        " train "
        print("-----Train-----")
        model.train()
        start_time = time.time()

        train_loss = train_per_epoch(
            model, 
            optimizer, 
            grad_accum_epoches, 
            train_dataloader
        )

        " valid "
        print("---Validation---")
        model.eval()

        ## output: predictions, reference and metrics for them
        predictions, references, metric = valid_per_epoch(
            model, 
            optimizer, 
            grad_accum_epoches, 
            valid_dataloader,
            metric
        )

        pr_list = {
            "preds": predictions,
            "refs": references
        }

        ## rouge/bleu score
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        print(result)


        ## save preds of validation set to check each epoch
        with open(f"./cache/preds_e{epoch+1}.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'predictions', 'references'])

            for pred, ref in zip(pr_list['preds'], pr_list['refs']):
                writer.writerow([pred, ref])
            print(f"prediction of validation set saved in ./cache/preds_{epoch}.csv!") 

        ## save and check
        model_path = os.path.join(
            "test/", "model_{}".format(epoch + 1))
        model.save_model(model_path)

        print("time:{}, epoch:{}/{}, train_loss:{}".format(
            time.time()-start_time, epoch+1, iteration, train_loss))

def main():
    " load data "
    if args.dataset_name is not None:
        ## Downloading and loading a dataset from the hub.
        # dataset = load_dataset("maderix/flickr_bw_rgb")
        datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file

        if args.validation_file is not None:
            data_files["valid"] = args.valid_file

        # if args.test_file is not None:
        #     data_files["test"] = args.test_file

        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    " dataset and dataloader"
    train_dataset = ImgCapDataset(raw_datasets["train"])
    valid_dataset = ImgCapDataset(raw_datasets["valid"])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        collate_fn=ImgCapDataset.collate_fn
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size,
        collate_fn=ImgCapDataset.collate_fn
    )

    training(train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()