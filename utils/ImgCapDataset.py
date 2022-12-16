from torch.utils.data import DataLoader, Dataset
import numpy as np

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