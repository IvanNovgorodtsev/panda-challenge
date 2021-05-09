import os
from skimage import io, transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import openslide

n_tiles = 16
tile_size = 256

def get_tiles(img, mode=0):
    
    result = []
    h, w, c = img.shape
    print(f'img.shape: {img.shape}')
    # number of "pixels" we need to pad both ways
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    # padded images, which can be divided each dimension by tile_size
    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]], constant_values=255)
    print(f'img2.shape: {img2.shape}')

    # getting the number of tiles in the padded image (getting the shape)
    img3 = img2.reshape(img2.shape[0] // tile_size, tile_size, img2.shape[1] // tile_size, tile_size, 3)
    img3.shape
    print(f'img3.shape: {img3.shape}')

    # reshaping image
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    print(f'img3.shape: {img3.shape}')

    # if number of tiles we prepared is lower than n_tiles defined, pad more
    print(f'len(img3):{len(img3)} n_tiles: {n_tiles}')
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    print(f'idxs: {idxs}')

    img3 = img3[idxs]
    print(f'img3.shape: {img3.shape}')

    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    time.sleep(60)
    return result

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

        image = openslide.OpenSlide(img_path)
        image = image.read_region((0,0), 2, image.level_dimensions[2])
        image = np.asarray(image)[:,:,0:3]
        print(img_path)
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
