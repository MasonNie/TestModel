import torch
from torch import nn
import numpy as np
from Ternery import *

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, )
        return y


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, blocksize=32):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.blocksize = blocksize

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # #bug聚集地
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(in_features=in_dim, out_features=in_dim*blocksize*blocksize,bias=True)
        # #end
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        '''bug聚集地'''
        # atten = self.avg_pool(out).view(m_batchsize,C)
        # atten = self.linear(atten)
        # atten = atten.view(m_batchsize, C, self.blocksize, self.blocksize)
        return out  # B * C * blocksize * blocksize


class csPart(nn.Module):
    def __init__(self, blocksize=32, subrate=0.2, channel=1):
        super(csPart, self).__init__()
        # sampling
        # self.sampling = Conv2d_Q(
        #     in_channels=channel,
        #     out_channels=int(np.round(blocksize * blocksize * subrate * channel)),
        #     kernel_size=blocksize,
        #     stride=blocksize,
        #     padding=0,
        #     bias=False,
        #     W=2)
        self.sampling = nn.Conv2d(
            in_channels=channel,
            out_channels=int(np.round(blocksize * blocksize * subrate * channel)),
            kernel_size=blocksize,
            stride=blocksize,
            padding=0,
            bias=False,)

        self.gamma = nn.Parameter(torch.zeros(1))
        # self.atten = SELayer(channel=int(np.round(blocksize*blocksize*subrate*channel), reduction=16))
        self.atten = Self_Attn(in_dim=int(np.round(blocksize * blocksize * subrate * channel)), activation=False,
                               blocksize=blocksize)  # SELayer(channel=int(np.round(blocksize*blocksize*subrate*channel)), reduction=16)
        # self.sampling = nn.Conv2d(1, int(np.round(blocksize*blocksize*subrate)), blocksize, stride=blocksize, padding=0, bias=False)
        # init reconstruction
        self.upsampling = nn.Conv2d(int(np.round(blocksize * blocksize * subrate * channel)), blocksize * blocksize, 1,
                                    stride=1, padding=0)
        # self.upsampling = nn.Sequential(
        #     nn.Conv2d(int(np.round(blocksize*blocksize*subrate*channel)), blocksize*blocksize, 1, stride=1, padding=0, bias=True),
        #     nn.LeakyReLU()
        # )

        self.dwc = nn.Sequential(
            nn.Conv2d(in_channels=blocksize * blocksize,
                      out_channels=blocksize * blocksize,
                      kernel_size=1, stride=1, padding=0,
                      groups=blocksize * blocksize),
            # nn.LeakyReLU(),
            nn.Conv2d(in_channels=blocksize * blocksize, out_channels=blocksize * blocksize, kernel_size=1, padding=0),
            # nn.LeakyReLU()
        )
        # self.activation = nn.LeakyReLU()

    def forward(self, input):
        x = self.sampling(input)
        x = self.gamma * x + x
        atten_matrix = self.atten(x)
        x = self.upsampling(atten_matrix)
        # x = self.activation(x)
        x = self.dwc(x)  # + x
        # x = self.activation(x)
        return x


class sconv2d(nn.Module):
    def __init__(self, channels=64, outchannels=64):
        super(sconv2d, self).__init__()
        self.channels = channels
        self.separable_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=(1, 1), stride=1, padding=0,
                      groups=channels),
            nn.Conv2d(in_channels=channels, out_channels=outchannels, kernel_size=(1, 1), padding=0),
            # nn.LeakyReLU()
        )
        # self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.separable_conv2d(input)
        # x = self.relu(x)
        return x


class baseblock(nn.Module):
    def __init__(self, channels=64):
        super(baseblock, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            sconv2d(channels=channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            sconv2d(channels=channels),
            nn.LeakyReLU(inplace=True)
        )
        # self.relu = nn.LeakyReLU()

    def forward(self, input):
        # x = self.relu(input)
        x = self.conv1(input)
        x = x + input
        x = self.conv2(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, conv_in, conv_out, k_size, beta=0.2):
        super(DenseBlock, self).__init__()

        self.res1 = nn.Sequential(
            # nn.Conv2d(conv_in, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            sconv2d(channels=conv_in, outchannels=conv_out),
            nn.LeakyReLU(inplace=True)
        )

        self.res2 = nn.Sequential(
            # nn.Conv2d(conv_in*2, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            sconv2d(channels=conv_in * 2, outchannels=conv_out),
            nn.LeakyReLU(inplace=True)
        )

        self.res3 = nn.Sequential(
            # nn.Conv2d(conv_in*3, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
            sconv2d(channels=conv_in * 3, outchannels=conv_out),
            # nn.LeakyReLU(inplace=True)
        )

        # self.res4 = nn.Sequential(
        # nn.Conv2d(conv_in*4, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
        # sconv2d(channels=conv_in*4, outchannels=conv_out),
        # nn.LeakyReLU(inplace=True),
        # )

        # self.res5 = nn.Sequential(
        #     nn.Conv2d(conv_in*5, conv_out, kernel_size=(k_size, k_size), stride=1, padding=1),
        #     sconv2d(channels=conv_in*3, outchannels=conv_out),
        # )
        self.beta = beta

    def forward(self, input):
        x = input
        # feature size = convin*2
        result = self.res1(x)
        x = torch.cat([x, result], 1)

        result = self.res2(x)
        # print(x.shape,result.shape)
        x = torch.cat([x, result], 1)

        x = self.res3(x)
        # x = torch.cat([x, result], 1)

        # x = self.res4(x)
        # x = torch.cat([x, result], 1)

        # x = self.res5(x)

        output = x.mul(self.beta)
        return output + input


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, conv_in=64, k_size=3, beta=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()

        self.dense1 = DenseBlock(conv_in, conv_in, k_size)
        self.dense2 = DenseBlock(conv_in, conv_in, k_size)
        self.dense3 = DenseBlock(conv_in, conv_in, k_size)
        self.beta = beta

    def forward(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        output = x.mul(self.beta)
        return output + input


class CSNetPlus(nn.Module):
    def __init__(self, blocksize=32, subrate=0.2, channels=1):
        super(CSNetPlus, self).__init__()
        self.blocksize = blocksize
        self.subrate = subrate
        self.channels = channels
        n_baseblock = 10
        outchannels = 64

        self.csPart = csPart(blocksize, subrate, channels)
        self.rrdb = ResidualInResidualDenseBlock(conv_in=outchannels, k_size=3, )
        self.reshape = nn.PixelShuffle(upscale_factor=self.blocksize)
        self.dalconv = nn.Sequential(
            nn.Conv2d(in_channels=outchannels,
                      out_channels=outchannels,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=2,
                      dilation=2,
                      bias=True),
            nn.LeakyReLU(inplace=True))

        self.dr1 = nn.Sequential(
            nn.Conv2d(channels, outchannels, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )
        baseblock_layers = []
        for i in range(n_baseblock):
            if (i % 2 == 1):
                baseblock_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=outchannels,
                              out_channels=outchannels,
                              kernel_size=(3, 3),
                              stride=1,
                              padding=2,
                              dilation=2,
                              bias=True),
                    nn.LeakyReLU(inplace=True)))

            # baseblock_layers.append(ResidualInResidualDenseBlock(conv_in=outchannels, k_size=3,))
            baseblock_layers.append(DenseBlock(conv_in=outchannels, conv_out=outchannels, k_size=3, beta=0.3))

        self.baseblock_seq = nn.Sequential(*baseblock_layers)

        self.out_conv = nn.Conv2d(outchannels, 1, kernel_size=(3, 3), stride=1, padding=1, bias=True)

    def forward(self, input):
        x = self.csPart(input)
        # x = self.rrdb(x)
        reshape = self.reshape(x)
        dr1 = self.dr1(reshape)
        # x = self.rrdb(dr1)
        # x = self.rrdb(x)
        x = self.baseblock_seq(dr1)
        x = self.out_conv(x)
        x = x + reshape
        return x, reshape

