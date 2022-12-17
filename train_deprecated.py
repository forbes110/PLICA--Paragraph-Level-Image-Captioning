import os
import time
import numpy as np
from datasets import load_dataset
from transformers import Adafactor
from torch.utils.data import DataLoader, Dataset

from model.ImgCaptionModel import ImgCaptionModel

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class ImgCaptionDataset(Dataset):
    def __init__(self, data):
        self.images = []
        self.captions = []
        for row in data:
            img = np.asarray(row["image"])
            if (len(img.shape) == 3):
                self.images.append(row["image"])
                self.captions.append(row["caption"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.captions[index]

    def collate_fn(self, samples):
        image_list = []
        caption_list = []
        for sample in samples:
            image_list.append(sample[0])
            caption_list.append(sample[1])

        return image_list, caption_list


def train_per_epoch(model, optimizer, batch_size, train_loader):
    train_loss = 0
    for idx, (images, captions) in enumerate(train_loader):
        print("\r", idx, end="")
        outputs = model(images, captions)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        if ((idx+1) % batch_size == 0):
            optimizer.step()
            optimizer.zero_grad()

    train_loss = train_loss / len(train_loader)
    return train_loss


def training(train_loader, val_loader):
    iteration = 10
    batch_size = 8
    model = ImgCaptionModel()

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

    print("Start training")
    for epoch in range(iteration):
        start_time = time.time()

        train_loss = train_per_epoch(
            model, optimizer, batch_size, train_loader)

        model_path = os.path.join(
            "test/", "model_{}".format(epoch + 1))
        model.save_model(model_path)

        print("time:{}, epoch:{}/{}, train_loss:{}".format(
            time.time()-start_time, epoch+1, iteration, train_loss))


def main():
    dataset = load_dataset("maderix/flickr_bw_rgb")

    ImgCapDataset = ImgCaptionDataset(dataset["train"])

    train_dataloader = DataLoader(ImgCapDataset, batch_size=4,
                            collate_fn=ImgCapDataset.collate_fn)
    
    training(train_dataloader, None)


if __name__ == '__main__':
    main()
