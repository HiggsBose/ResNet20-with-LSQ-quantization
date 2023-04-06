import ResidualBlock
import torch
import torch.nn as nn
import configuration as cfg


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # the model structure is arbitrarily set by me based on ResNet, with more fine-tune I think the accuracy will rise
        self.layer1 = ResidualBlock.Conv(1, 8)
        self.layer2 = nn.Sequential(
            ResidualBlock.ResidualBlock(8, 16, stride=1, padding=1),
            ResidualBlock.ResidualBlock(16, 16),
            ResidualBlock.ResidualBlock(16, 32, stride=2, padding=1),
            ResidualBlock.ResidualBlock(32, 32),
            ResidualBlock.ResidualBlock(32, 64, stride=2, padding=1),
            ResidualBlock.ResidualBlock(64, 64),
            ResidualBlock.ResidualBlock(64, 128),
            ResidualBlock.ResidualBlock(128, 128)
        )

        self.layer3 = ResidualBlock.Linear(7*7*128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out
