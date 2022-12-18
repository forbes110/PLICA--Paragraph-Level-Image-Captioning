from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image

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


        self.call_img = {
            'image': row[image_name],
            'image_path': Image.open(image_name)
        }

        " init column names, for flicker and visual_genome"
        if 'image' in data[0]:
            image_name = 'image'
            caption_name = 'caption'

        elif 'image_path' in data[0]:
            image_name = 'image_path'
            caption_name = 'paragraph'
        
        else:
            print('No column match! dataset column names need to be checked!')

        for row in data:

            " need to resize to 256*256 "
            # img = row[image_name].resize((256,256))
            # img = self.transform(row[image_name])
            img = self.transform(self.call_img[image_name])

            # img = img.resize((256,256))
            # print('img', img)

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



# from torch.utils.data import DataLoader, Dataset
# import numpy as np

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