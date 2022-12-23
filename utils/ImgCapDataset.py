from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm


class ImgCapDataset(Dataset):
    '''
        need to check no captions mode for preict,
        and where to read the image data    

        return (id, image, caption)

        data form:
        [
            {
                "url": "https://cs.stanford.edu/people/rak248/VG_100K/2317429.jpg",
                "id": 2317429,
                "paragraph": "A white round plate is on a table with a plastic tablecloth on it.  Two foil covered food halves are on the white plate along with a serving of golden yellow french fries.  Next to the white plate in a short,  topless, plastic container is a white sauce.  Diagonal to the white plate are the edges of several other stacked plates.  There are black shadows reflected on the table.",
                "image_path": "./data/processed/img/2317429.jpg"
            },
        ]
    '''
    def __init__(self, data):

        self.ids = []
        self.images = []
        self.captions = []

        ## standarize the size
        self.transform = T.Resize((256,256))

        id_name = list(data.features.keys())[1]
        caption_name = list(data.features.keys())[2]
        image_name = list(data.features.keys())[3]

        print('the image_name input:', image_name)
        print('the caption_name input:', caption_name)

        load_bar = tqdm(data, total=len(data), position=0, leave=True, ncols=100)
        for row in load_bar:
            ## need to resize to 256*256
            img = self.transform(Image.open(row[image_name]))

            ## check dims are A* B* C with numpy
            img_np = np.asarray(img)
            if (len(img_np.shape) == 3):
                self.ids.append(row[id_name])
                self.images.append(img)
                self.captions.append(row[caption_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.ids[index], self.images[index], self.captions[index]

    def collate_fn(self, samples):
        ids = []
        image_list = []
        caption_list = []

        for sample in samples:
            ids.append(sample[0])
            image_list.append(sample[1])
            caption_list.append(sample[2])

        return ids, image_list, caption_list

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




