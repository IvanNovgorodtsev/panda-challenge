import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchsummary import summary

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
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        ## transposed convolutions
        # in_channels, out_channels, kernel_size, stride
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_4 = double_conv(128, 64)

        # for multiple object classification increase the number of output channels
        self.out = nn.Conv2d(64, 3, 1)


    def forward(self, image):
        # batch size, channel, hight, width
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    #summary(model, input_size=(3, 128, 128))

    image = load_image('data/images/image_0.jpg')
    mask = load_image('data/masks/mask_0.jpg')

    image = np.expand_dims(image,0)
    image = image.transpose(0,3,1,2)
    input_data = torch.tensor(image).to(device)

    #mask = np.expand_dims(mask, 0)
    mask = mask.transpose(2, 0, 1)
    mask_data = torch.tensor(mask)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    y_true = torch.argmax(mask_data, dim=0)
    y_true = y_true.unsqueeze(0)

    for t in range(100):
        y_pred = model(input_data)
        loss = criterion(y_pred, y_true)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


