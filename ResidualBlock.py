# basic building blocks of ResNet

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel):
        '''
        this is the first convolution of the model

        :param input_channel: the input channel should be 1 because the FashionMNIST dataset contains images with only 1
        channel
        :param output_channel: it can be anything you want
        '''
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU())

    def forward(self, x):
        out = self.layer(x)
        # print(out.size())
        return out


class ResidualBlock(nn.Module):

    def __init__(self, input_channel, output_channel, stride=1, padding=1):
        '''
        Residual block for ResNet, it contains two convolution layers and one shortcut to enable residual learning
        :param input_channel: the input channel should be the same as the former layer's output channel
        :param output_channel: anything you want
        :param stride: the stride of the kernel in the first convolution layer inside the residual block
        :param padding: the corresponding padding number
        '''
        super(ResidualBlock, self).__init__()
        self.ConvUnit = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel))
        self.relu = nn.ReLU()
        # if the input and the output size of the Residual Block does not match, use a shortcut with a 1*1 convolution
        # to adjust the sizes
        if stride != 1 or input_channel != output_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel))
        else:
            self.residual = nn.Sequential()

    def forward(self, x):
        out = self.ConvUnit(x)
        # print(out.size())
        # print(self.residual(x).size())
        # print(x.size())
        out = out + self.residual(x)               # residual connection
        out = self.relu(out)

        return out


class Linear(nn.Module):
    def __init__(self, input_neuron, output_neuron):
        super(Linear, self).__init__()
        self.layer = nn.Linear(input_neuron, output_neuron)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.soft_max(out)
        return out

