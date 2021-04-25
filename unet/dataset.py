import os
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import time

class Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_dir = images_dir
        self.mask_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = io.imread(img_path)
        image = np.asarray(image, dtype=np.float32)
        mask = np.load(mask_path)

        # loading images as 3 channel RGB
        image = image.transpose(2, 0, 1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
