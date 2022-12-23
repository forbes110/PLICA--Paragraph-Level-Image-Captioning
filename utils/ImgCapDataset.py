from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm


class ImgCapDataset(Dataset):
    '''
        need to check no captions mode for preict,
        and where to read the image data    
    '''
    def __init__(self, data):


        self.images = []
        self.captions = []

        ## standarize the size
        self.transform = T.Resize((256,256))

        caption_name = list(data.features.keys())[2]
        image_name = list(data.features.keys())[3]

        print('the image_name input:', image_name)
        print('the caption_name input:', caption_name)

        train_bar = tqdm(data, total=len(data), position=0, leave=True, ncols=100)
        for row in train_bar:
            ## need to resize to 256*256
            img = self.transform(Image.open(row[image_name]))

            ## check dims are A* B* C with numpy
            img_np = np.asarray(img)
            if (len(img_np.shape) == 3):
                self.images.append(img)
                self.captions.append(row[caption_name])

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

'''
    for flicker test
'''
# class ImgCapDataset(Dataset):
#     def __init__(self, data):
#         self.images = []
#         self.captions = []
#         for row in data:
#             img = np.asarray(row["image"])
#             if (len(img.shape) == 3):
#                 self.images.append(row["image"])
#                 self.captions.append(row["caption"])

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         return self.images[index], self.captions[index]

#     def collate_fn(self, samples):
#         image_list = []
#         caption_list = []
#         for sample in samples:
#             image_list.append(sample[0])
#             caption_list.append(sample[1])

#         return image_list, caption_list




