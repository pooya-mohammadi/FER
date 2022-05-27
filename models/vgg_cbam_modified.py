import torch
from torch.nn import functional as F
from torch import nn
from models.cbam_modified import CBAM


class VggCBAM(nn.Module):
    def __init__(self, drop=0.3, crop=40, **kwargs):
        super().__init__()
        self.crop = crop
        self.conv_bn_prelu_1a = ConvBNPRELU(1, 64)
        self.conv_bn_prelu_1b = ConvBNPRELU(64, 64)

        self.conv_bn_prelu_2a = ConvBNPRELU(64, 128)
        self.conv_bn_prelu_2b = ConvBNPRELU(128, 128)

        self.conv_bn_prelu_3a = ConvBNPRELU(128, 256)
        self.conv_bn_prelu_3b = ConvBNPRELU(256, 256)

        self.conv_bn_prelu_4a = ConvBNPRELU(256, 512)
        self.conv_bn_prelu_4b = ConvBNPRELU(512, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(512 * (9 if self.crop == 48 else 4), 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.drop = nn.Dropout(p=drop)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x, get_cbam=False):
        x = self.conv_bn_prelu_1a(x)
        x = self.conv_bn_prelu_1b(x)
        before_bam1 = x
        after_bam1 = self.cbam1(before_bam1)
        x = self.pool(after_bam1)

        x = self.conv_bn_prelu_2a(x)
        x = self.conv_bn_prelu_2b(x)
        before_bam2 = x
        after_bam2 = self.cbam2(before_bam2)
        x = self.pool(after_bam2)

        x = self.conv_bn_prelu_3a(x)
        x = self.conv_bn_prelu_3b(x)
        before_bam3 = x
        after_bam3 = self.cbam3(before_bam3)
        x = self.pool(after_bam3)

        x = self.conv_bn_prelu_4a(x)
        x = self.conv_bn_prelu_4b(x)
        before_bam4 = x
        after_bam4 = self.cbam4(before_bam4)
        x = self.pool(after_bam4)

        x = x.view(-1, 512 * (9 if self.crop == 48 else 4))
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))
        x = self.classifier(x)
        if get_cbam:
            return x, [(before_bam1, after_bam1), (before_bam2, after_bam2), (before_bam3, after_bam3)]
        return x


class ConvBNPRELU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.prelu = nn.ReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


if __name__ == '__main__':
    from torchsummary import summary

    crop = 40
    model = VggCBAM(crop=crop).to('cuda')
    model(torch.zeros((1, 1, crop, crop)).to('cuda'))
    summary(model, (1, crop, crop))
