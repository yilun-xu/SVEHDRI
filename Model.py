import torch
import torch.nn as nn
import torch.nn.functional as F

class SVC8_5x5(nn.Module):
    def __init__(self, in_nc=1, out_nc=64):
        super(SVC8_5x5, self).__init__()

        self.pad = nn.ZeroPad2d(padding=(2, 2, 2, 2))
        self.conv_first1 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first2 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first3 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first4 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first5 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first6 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first7 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)
        self.conv_first8 = nn.Conv2d(in_nc, out_nc, 5, (4,2), 0, bias=True)

    def forward(self, x):
        input = self.pad(x)
        fea1 = self.conv_first1(input[:,:,0:,0:])
        fea2 = self.conv_first2(input[:,:,0:,1:])
        fea3 = self.conv_first3(input[:,:,1:,0:])
        fea4 = self.conv_first4(input[:,:,1:,1:])
        fea5 = self.conv_first5(input[:,:,2:,0:])
        fea6 = self.conv_first6(input[:,:,2:,1:])
        fea7 = self.conv_first7(input[:,:,3:,0:])
        fea8 = self.conv_first8(input[:,:,3:,1:])

        fea = torch.zeros((fea1.size(0), fea1.size(1), x.size(2), x.size(3)))
        fea[:, :, 0::4, 0::2] = fea1
        fea[:, :, 0::4, 1::2] = fea2
        fea[:, :, 1::4, 0::2] = fea3
        fea[:, :, 1::4, 1::2] = fea4
        fea[:, :, 2::4, 0::2] = fea5
        fea[:, :, 2::4, 1::2] = fea6
        fea[:, :, 3::4, 0::2] = fea7
        fea[:, :, 3::4, 1::2] = fea8
        return fea

class ResBlock_ComBlock(nn.Module):
    def __init__(self):
        super(ResBlock_ComBlock, self).__init__()

        self.conv0_res = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_res = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv0_com = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv1_com = nn.Conv2d(32, 64, 3, 1, 1)

        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):

        fea_com = self.conv1_com(F.leaky_relu(self.conv0_com(x[1]), 0.1, inplace=True))
        fea_rea = self.conv1_res(F.relu(self.conv0_res(x[0]), inplace=True)) + x[0]

        fea_rea = fea_rea + self.beta * fea_com

        return (fea_rea, fea_com)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        in_nc = 1 # Number of input channels
        out_nc = 3 # Number of output channels
        nf = 64 # Number of feature channels
        nb = 16 # Number of blocks

        self.conv_first_Res = SVC8_5x5(in_nc, nf)

        self.conv_first_Com = nn.Sequential(
            SVC8_5x5(in_nc, nf),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        double_branch = []
        for i in range(nb):
            double_branch.append(ResBlock_ComBlock())
        self.sft_branch = nn.Sequential(*double_branch)

        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True),
        )

    def forward(self, Bayer_Radiance_map, Exposure_Guidance_Mask):
        # Bayer_Radiance_map'shape : batch_size X 1 X Height X Wdith
        # Exposure_Guidance_Mask'shape : batch_size X 1 X Height X Wdith
        fuse = Bayer_Radiance_map * Exposure_Guidance_Mask

        fea_res = self.conv_first_Res(Bayer_Radiance_map)
        fea_com = self.conv_first_Com(fuse)

        fea = self.sft_branch((fea_res, fea_com))[0]

        out = self.conv_last(fea)

        return out

