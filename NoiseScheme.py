# adding noise to the model
# 7/28/2022

import torch
import torch.nn as nn
import torch.nn.functional as F
import configuration as cfg


def conv_fn(m_of_dispersion, additive=True, eval=False):

    class Conv2d(nn.Conv2d):
        def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0,
                     dilation=1, groups=1, bias=False):
            super(Conv2d, self).__init__(in_channel, out_channel, kernel_size, stride,
                                         padding, dilation, groups, bias)

        def forward(self, input):
            if self.training:
                if eval:
                    return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)
                else:
                    weight_n = add_noise(self.weight, m_of_dispersion=m_of_dispersion, additive=additive)
                    return F.conv2d(input, weight_n, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)
            else:
                if cfg.config["inference_noise"]:
                    weight_n = add_noise(self.weight, m_of_dispersion=m_of_dispersion, additive=additive)
                    return F.conv2d(input, weight_n, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)
                else:
                    return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups)
    return Conv2d


def add_noise(weight, m_of_dispersion, additive=True):

    std_deviation = weight.max() * m_of_dispersion
    random_noise = torch.randn(weight.size()).to(cfg.config["device"]) * std_deviation
    if additive:
        out_weight = weight + random_noise
    else:
        out_weight = weight * (torch.ones(weight.size()).to(cfg.config["device"]) + random_noise)

    return out_weight




