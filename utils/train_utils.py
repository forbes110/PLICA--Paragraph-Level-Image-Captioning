import csv
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from utils.metrics.bleu import bleu_score
from utils.metrics.cider import cider_score
from utils.metrics.meteor import meteor_score
from typing import *
from dataclasses import dataclass, field
import wandb


def save_preds(epoch, val_ids, pr_list):
    '''
        save preds/refs to compare
    '''
    with open(f"./cache/preds_e{epoch+1}.csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'predictions', 'references'])

        for val_id, pred, ref in zip(val_ids, pr_list['preds'], pr_list['refs']):
            writer.writerow([val_id, pred, ref])
        print(
            f"prediction of validation set saved in ./cache/preds_{epoch+1}.csv!")

@dataclass
class Metrics():
    predictions:Dict[int,str] = field(default_factory=lambda:{})
    references:Dict[int,str] = field(default_factory=lambda:{})
    scores:Dict[str,float] = field(default_factory=lambda:{})

    def add_batch(self, ids:List[int], preds:List[str], refs:List[str]=None):
        if refs is not None:
            for (id, pred, ref) in zip(ids, preds, refs):
                self.predictions[id] = [pred]
                self.references[id] = [ref]
        else:
            for (id, pred) in zip(ids, preds):
                self.predictions[id] = [pred]

    def compute(self):
        scores = {}
        b_s = bleu_score(self.references, self.predictions)
        for i in b_s.keys():
            scores[i] = b_s[i]
        c_s = cider_score(self.references, self.predictions)
        scores['CIDEr'] = c_s['CIDEr']
        m_s = meteor_score(self.references, self.predictions)
        scores['METEOR'] = m_s['METEOR']
        return scores


def train_per_epoch(model, optimizer, lr_scheduler, train_dataloader, epoch, args):
    '''
        for each batch rounds in a epoch
    '''
    train_loss = 0

    train_bar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=False, ncols=100)
    train_bar.set_description('Train')


    for idx, (_, images, captions) in enumerate(train_bar):
        outputs = model(images, captions)
        loss = outputs.loss
        train_bar.set_postfix({'loss': loss.detach().item()})

        train_loss += loss.item()
        loss.backward()
        wandb.log({"train_loss": train_loss})


        ## L1-regularization
        if args.use_L1reg == True:

            ## L1_reg collect from all params
            L1_reg = torch.tensor(0.).to(args.device)
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param).to(args.device))
                
            ## L1 penalty
            loss +=  args.lambda_val * L1_reg

        if ((idx+1) % args.grad_accu_step == 0):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    train_loss = train_loss / len(train_dataloader)

    return train_loss


def valid_per_epoch(model, valid_dataloader, gen_kwargs):

    # to save result pair
    metric = Metrics()
    predictions, references = [], []
    valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader), position=0, leave=True, ncols=100)
    valid_bar.set_description('Validation')


    for ids, images, captions in valid_bar:

        with torch.no_grad():
            predictions += model.inference(images, gen_kwargs)
            references += captions

            metric.add_batch(
                ids=ids,
                preds=predictions,
                refs=references,
            )
    scores = metric.compute()

    return ids, predictions, references, scores


def predict(model, test_dataloader, gen_kwargs):

    # to save result pair
    metric = Metrics()
    predictions, references = [], []
    test_bar = tqdm(test_dataloader, total=len(test_dataloader), position=0, leave=True, ncols=100)
    test_bar.set_description(f'Predict: ')


    for ids, images, captions in test_bar:

        with torch.no_grad():
            predictions += model.inference(images, gen_kwargs)
            references += captions

            metric.add_batch(
                ids=ids,
                preds=predictions,
            )
    scores = metric.compute()

    return ids, predictions, references, scores

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

        if args.test_file is not None:
            data_files["test"] = args.test_file

        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    return raw_datasets


