import os
from skimage import io, transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import time

n_tiles = 16
tile_size = 256

def get_tiles(img, mode=0):
    
    result = []
    h, w, c = img.shape

    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)
    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],constant_values=255)
    print(h,w,c)
    time.sleep(60)
    return 0

class Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, device, transform=None):
        self.image_dir = images_dir
        self.mask_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = io.imread(img_path)
        image = np.asarray(image, dtype=np.float32)
        mask = io.imread(mask_path)

        tiles = get_tiles(image)


        # loading images as 3 channel RGB
        image = image.transpose(2, 0, 1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = torch.tensor(image).to(self.device)
        mask = torch.tensor(mask).to(self.device)

        return image, mask
