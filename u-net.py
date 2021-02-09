import numpy as np
import os
import openslide
import time
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp
import torch
from skimage.transform import resize


class Dataset(BaseDataset):
    def __init__(self, images_path, masks_path, df_path):
        

        df = pd.read_csv(df_path)
        masks = os.listdir(masks_path)
        masks_df = pd.Series(masks).to_frame()
        masks_df.columns = ['mask_file_name']
        masks_df['image_id'] = masks_df.mask_file_name.apply(lambda x: x.split('_')[0])
        df = pd.merge(df, masks_df, on='image_id', how='outer')
        print(f"There are {len(df[df.mask_file_name.isna()])} images without a mask.")
        df = df[~df.mask_file_name.isna()]
        self.df = df


        self.dff = self.df.loc[self.df['data_provider'] == 'radboud']
        self.ids = self.dff.image_id

        self.im_path = [os.path.join(images_path, image_id + '.tiff') for image_id in self.ids]
        self.mask_path = [os.path.join(masks_path, mask_id+ '_mask.tiff') for mask_id in self.ids]

    def __getitem__(self, i):
        #print(self.im_path[i])
        im = openslide.OpenSlide(self.im_path[i])
        im_data = im.read_region((0, 0), im.level_count - 1, im.level_dimensions[-1])
        data = np.asarray(im_data)[:, :, 0:3]
        data = resize(data, (256, 256))
        data = np.moveaxis(data, -1, 0)
        data = data.astype(np.float32)

        #print(self.mask_path[i])
        mask = openslide.OpenSlide(self.mask_path[i])
        mask_data = mask.read_region((0, 0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = np.asarray(mask_data)[:, :, 0]
        mask_data = resize(mask_data, (256, 256))

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # data, mask_data = data.to(device), mask_data.to(device)
        return data, mask_data
        #reading images and masks

    def __len__(self):
        return len(self.ids)

    def lis(self):
        return self.mask_path


# sprawdzenie czy dane są ładowane poprawnie
# data = Dataset('train','masks','train-2.csv')
# data, mask = data.__getitem__(6)
# print(data[100][150:200])
# print(mask[100][150:200])
# print(data.shape, mask.shape)
# x = np.moveaxis(data, -1, 0)
# print(x.shape)


ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['0','1','2','3','4','5']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# # create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

dataset = Dataset('/mnt/gpudata1/prostate-cancer-grade-assessment/train_images','/mnt/gpudata1/prostate-cancer-grade-assessment/train_label_masks','train.csv')
batch_size = 6
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
for i in range(0, 10):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(validation_loader)

