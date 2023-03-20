# Attempts to quantize the resnet model
# 7/27/2022


from torch import nn
from torch.nn import functional as F
import QuantizationScheme as Q
import configuration as cfg
import NoiseScheme as N


# define the Residual block structure
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        self.ConvUnit = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()  # if input and output are of the same dimension, use the identity mapping shortcut
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )   # if input and output differ in dimension, use 1*1 convolution to adjust shortcut dimensions to match the size

    def forward(self, x):
        out = self.ConvUnit(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# quantized block
class QuantizeResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(QuantizeResidualBlock, self).__init__()
        Conv2d = Q.conv2d_Q_fn(cfg.config["w_bit"])
        self.require_vis = False
        self.conv1 = Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel)
            )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        quantize = Q.LsqQuan(cfg.config['w_bit'], all_positive=True, is_activation=True)
        x = quantize(x)
        out = self.conv1(x)
        out1 = self.bn1(out)
        out2 = self.relu(out1)
        out2 = quantize(out2)
        out3 = self.conv2(out2)
        out4 = self.bn2(out3)
        out5 = out4 + self.shortcut(x)
        out6 = F.relu(out5)
        if self.require_vis:
            data = {"initial": x.view(1, -1).to('cpu').detach(), "after_conv1": out.view(1, -1).to('cpu').detach(),
                    "after_bn1": out1.view(1, -1).to('cpu').detach(), "after_relu": out2.view(1, -1).to('cpu').detach(),
                    "after_conv2": out3.view(1, -1).to('cpu').detach(),"after_bn2": out4.view(1, -1).to('cpu').detach(),
                    "after_shortcut": out5.view(1, -1).to('cpu').detach(), "out": out6.view(1, -1).to('cpu').detach()}
            return out6, data
        else:
            return out6


# block with noise
class NoisedResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(NoisedResidualBlock, self).__init__()
        Conv2d = N.conv_fn(cfg.config["m_of_dispersion"], cfg.config["additive"], eval=False)
        self.conv1 = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
