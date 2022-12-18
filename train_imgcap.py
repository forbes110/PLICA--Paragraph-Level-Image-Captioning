import os
import time
import evaluate
from transformers import Adafactor

from torch.utils.data import DataLoader
from config import parse_args
from tqdm.auto import tqdm

from model.ImgCapModel import ImgCapModel
from utils.ImgCapDataset import ImgCapDataset
from utils.train_utils import (
    save_preds,
    train_per_epoch,
    valid_per_epoch,
    load_raw_datasets
)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


'''
    configs
'''
args = parse_args()


def training(train_dataloader, valid_dataloader):
    '''
        main function of train
    '''
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

    # to save results
    print("***Start training***")

    for epoch in range(args.num_epoches):
        print(f"\n=== Epoch {epoch+1} ===")

        print("-----Train-----")
        model.train()
        start_time = time.time()

        # per batch
        train_loss = train_per_epoch(
            model,
            optimizer,
            args.grad_accu_step,
            train_dataloader
        )

        # print("---Validation---")
        # model.eval()
        # gen_kwargs = {
        #     "max_length": args.val_max_target_length,
        #     "num_beams": args.num_beams,
        #     "do_sample": args.do_sample,
        #     "top_k": args.top_k,
        #     "top_p": args.top_p,
        #     "typical_p": args.typical_p,
        #     "temperature": args.temperature,
        #     "repetition_penalty": args.repetition_penalty,
        #     "no_repeat_ngram_size": args.no_repeat_ngram_size,
        # }

        # # output: predictions, reference and metrics for them
        # predictions, references, metric = valid_per_epoch(
        #     model,
        #     optimizer,
        #     args.grad_accu_step,
        #     valid_dataloader,
        #     metric,
        #     gen_kwargs
        # )

        # pr_list = {
        #     "preds": predictions,
        #     "refs": references
        # }

        # # rouge/bleu score
        # result = metric.compute(use_stemmer=True)
        # result = {k: round(v * 100, 4) for k, v in result.items()}
        # print(result)

        # save preds of validation set to check each epoch
        # save_preds(epoch, pr_list)

        # save and check
        model_path = os.path.join(
            "test/", "model_{}".format(epoch + 1))
        model.save_model(model_path)

        print("time:{}, epoch:{}/{}, train_loss:{}".format(
            time.time()-start_time, epoch+1, args.num_epoches, train_loss))


def main():
    " load data "
    raw_datasets = load_raw_datasets(args)

    " dataset and dataloader"
    train_dataset = ImgCapDataset(raw_datasets["train"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = ImgCapDataset(raw_datasets["valid"])
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_dataset.collate_fn
    )

    "train/valid"
    training(train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()
