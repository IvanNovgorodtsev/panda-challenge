import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchsummary import summary
import time
from unet.dataset import *

def load_image(infilename):
    img = Image.open(infilename, 'r')
    img.load()
    data = np.asarray(img, dtype="float32")
    return data


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    # square images
    target_size = target_tensor.size()[2]
    if tensor.size()[2] % 2 == 1:
        tensor_size = tensor.size()[2]-1
    else:
        tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self, nb_classes):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        ## transposed convolutions
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, nb_classes, 1)


    def forward(self, image):
        # encoder part
        # input image
        x1 = self.down_conv_1(image) # this is passed to decoder
        # max pooling
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv_2(x2) # this is passed to decoder
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv_3(x4) # this is passed to decoder
        x6 = self.max_pool_2x2(x5)

        x7 = self.down_conv_4(x6) # this is passed to decoder
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv_5(x8)

        # decoder part
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)
        return x


if __name__ == "__main__":

    batch_size = 3
    num_epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'DEVICE: {device}')

    dataset = Dataset('/kaggle/input/prostate-cancer-grade-assessment/train_images', '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks', device)
    dataset_size = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = UNet(nb_classes=6)
    model = model.to(device)
    # sparcecategoricalentropy for number in targets
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # #summary(model, input_size=(3, 128, 128))

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # labels = labels.squeeze()
            y_pred = model(inputs)
            # print(y_pred.shape)
            loss = criterion(y_pred, labels)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, i: {i}, loss: {loss.item()}')

    model.eval()
    image = load_image('data/images/image_1999.jpg')
    image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    input_data = torch.tensor(image).to(device)

    pred = model(input_data)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)
    np.save('pred', pred)



