import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2




def load_image(infilename):
    """This function loads an image into memory when you give it
       the path of the image
    """
    img = Image.open(infilename, 'r')
    img.load()
    data = np.asarray(img, dtype="float32")
    return data


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        ## transposed convolutions
        # in_channels, out_channels, kernel_size, stride
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 3, 2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.up_conv_4 = double_conv(128, 64)

        # for multiple object classification increase the number of output channels
        self.out = nn.Conv2d(64, 2, 1)


    def forward(self, image):
        # bs, channel, hight, width
        # encoder part

        # input image
        x1 = self.down_conv_1(image) # this part is passed to decoder

        # max pooling
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv_2(x2) # this part is passed to decoder
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv_3(x4) # this part is passed to decoder
        x6 = self.max_pool_2x2(x5)

        x7 = self.down_conv_4(x6) # this part is passed to decoder
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv_5(x8)

        # decoder part
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 3))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 3))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 3))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 3))

        x = self.out(x)
        return x

if __name__ == "__main__":
    image = load_image('data/images/image_0.jpg')
    image = np.expand_dims(image,0)
    image = image.transpose((0,3,1,2))
    input_data = torch.tensor(image)
    print(input_data.shape)
    model = UNet()
    model.forward(input_data)


