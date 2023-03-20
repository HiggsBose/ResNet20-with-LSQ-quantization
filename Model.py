
from torch import nn
from torch.nn import functional as F
import BuildingBlock as B
import configuration as cfg


# define the model structure
class ResNet(nn.Module):
    def __init__(self, residual_block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(residual_block, 16, 3, stride=1)
        self.layer2 = self.make_layer(residual_block, 32, 3, stride=2)
        self.layer3 = self.make_layer(residual_block, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, bias=False):
        strides = [stride] + [1] * (num_blocks - 1)  # strides = [1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# create the model
def ResNet20():
    '''
    construct the untrained network

    :return: pristine network on 'device'
    '''

    # quantization = int(input('Choose to quantize the model or not? (1 for yes and 0 for no): '))
    # print('your choice is', quantization)
    # noise = int(input('Choose to add noise or not? (1 for yes and 0 for no): '))
    # print('your choice is', noise)
    quantization = 0
    noise = 0

    if quantization:
        if not noise:
            print('model is quantized, and there is no noise')
            return ResNet(B.QuantizeResidualBlock).to(cfg.config["device"])
        else:
            print('model is quantized with noise')
            cfg.config["add_noise"] = True
            return ResNet(B.QuantizeResidualBlock).to(cfg.config["device"])
    else:
        if not noise:
            print('this is the basic model')
            return ResNet(B.ResidualBlock).to(cfg.config["device"])
        else:
            print('model is injected with noise but not quantized')
            return ResNet(B.NoisedResidualBlock).to(cfg.config["device"])