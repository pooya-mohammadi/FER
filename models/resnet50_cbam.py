import torch
from models.cbam import CBAM
import torch.nn as nn
from torchsummary import summary
from .resnet50 import Resnet


filters = [64, 128, 256, 512]


class CbamBottleNeck(nn.Module):
    expantion = 4

    def __init__(self, in_channel, out_channel, transition=None, stride=1, **kwargs):
        super(CbamBottleNeck, self).__init__()
        # self.cbam_blocks = kwargs['cbam_blocks']
        if 'residual_cbam' in kwargs:
            self.residual_cbam = kwargs["residual_cbam"]
        else:
            self.residual_cbam = False
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, self.expantion * out_channel, kernel_size=1, stride=1, bias=False)
        self.BN3 = nn.BatchNorm2d(self.expantion * out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.transition = transition
        self.CBAM = CBAM(self.expantion * out_channel, self.residual_cbam)

    def forward(self, x):
        y = self.conv1(x)
        y = self.BN1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.BN2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.BN3(y)
        y = self.CBAM(y)
        if self.transition is not None:
            x = self.transition(x)

        y = self.relu(x + y)
        return y


if __name__ == '__main__':
    from torchsummary import summary

    model = Resnet(CbamBottleNeck, 7, [3, 4, 6, 3]).to('cuda')
    model(torch.zeros((1, 1, 40, 40)).to('cuda'))
    summary(model, (1, 48, 48))
