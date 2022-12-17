import csv
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def save_preds(epoch, pr_list):
    '''
        save preds/refs to compare
    '''
    with open(f"./cache/preds_e{epoch+1}.csv", "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'predictions', 'references'])

        for pred, ref in zip(pr_list['preds'], pr_list['refs']):
            writer.writerow([pred, ref])
        print(f"prediction of validation set saved in ./cache/preds_{epoch}.csv!") 


def train_per_epoch(model, optimizer, grad_accum_step, train_dataloader):
    '''
        for each batch rounds in a epoch
    '''
    train_loss = 0

    train_bar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=True, ncols=100)

    for idx, (images, captions) in enumerate(train_bar):
        print("\r", idx, end="")
        outputs = model(images, captions)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        if ((idx+1) % grad_accum_step == 0):
            optimizer.step()
            optimizer.zero_grad()

    train_loss = train_loss / len(train_per_epoch)
    return train_loss


def valid_per_epoch(model, optimizer, batch_size, valid_dataloader, metric, gen_kwargs):

    ## to save result pair
    predictions, references = [], []
    valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader), position=0, leave=True, ncols=100)
    for idx, (images, captions) in enumerate(valid_bar):

        with torch.no_grad():
            # ...

            decoded_preds, decoded_labels = 0, 0
            predictions, references = 0, 0
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )

        return predictions, references, metric


def load_raw_datasets(args):
    '''
        load train/valid/test datasets
    '''
    if args.dataset_name is not None:
        ## Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name)
        
        # Deal with datasets with no validation set
        if args.dataset_name == "maderix/flickr_bw_rgb":
            raw_datasets["valid"] = []

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

    return raw_datasets