import csv
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from pycocoevalcap.meteor.meteor import Meteor 
from metrics import (
    bleu_score, cider_score
)


def save_preds(epoch, pr_list):
    '''
        save preds/refs to compare
    '''
    with open(f"./cache/preds_e{epoch+1}.csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'predictions', 'references'])

        for pred, ref in zip(pr_list['preds'], pr_list['refs']):
            writer.writerow([pred, ref])
        print(
            f"prediction of validation set saved in ./cache/preds_{epoch+1}.csv!")


def train_per_epoch(model, optimizer, lr_scheduler, grad_accum_step, train_dataloader, epoch, args):
    '''
        for each batch rounds in a epoch
    '''
    train_loss = 0

    train_bar = tqdm(train_dataloader, total=len(
        train_dataloader), position=0, leave=False, ncols=100)

    for idx, (images, captions) in enumerate(train_bar):
        outputs = model(images, captions)
        loss = outputs.loss
        train_bar.set_postfix({'loss': loss.detach().item()})

        train_loss += loss.item()
        loss.backward()

        ## L1-regularization
        if args.use_L1reg == True:

            ## L1_reg collect from all params
            L1_reg = torch.tensor(0.).to(args.device)
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param).to(args.device))
                
            ## L1 penalty
            loss +=  args.lambda_val * L1_reg

        train_bar.set_description(f'Epoch [{epoch+1}/{args.num_epoches}]')
        if ((idx+1) % grad_accum_step == 0):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    train_loss = train_loss / len(train_dataloader)

    return train_loss


def valid_per_epoch(model, optimizer, batch_size, valid_dataloader, metric, gen_kwargs):

    # to save result pair
    predictions, references = [], []
    valid_bar = tqdm(valid_dataloader, total=len(
        valid_dataloader), position=0, leave=True, ncols=100)
    for idx, (images, captions) in enumerate(valid_bar):

        with torch.no_grad():
            predictions += model.inference(images, gen_kwargs)
            references += captions

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

    return predictions, references, metric


def load_raw_datasets(args):
    '''
        load train/valid/test datasets
    '''
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name)

        # Deal with datasets with no validation set
        if args.dataset_name == "maderix/flickr_bw_rgb":
            print('the valid dataset is empty!')
            raw_datasets["valid"] = []

    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file

        if args.valid_file is not None:
            data_files["valid"] = args.valid_file

        # if args.test_file is not None:
        #     data_files["test"] = args.test_file

        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    return raw_datasets

class metric:
    def add_batch():
        pass