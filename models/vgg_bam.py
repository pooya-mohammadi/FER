import torch
from torch.nn import functional as F
from torch import nn
from models.bam import BAM


class VggBAM(nn.Module):
    def __init__(self, drop=0.2, crop=40, **kwargs):
        super().__init__()
        self.crop = crop
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512 * (9 if self.crop == 48 else 4), 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.bam1 = BAM(128)
        self.bam2 = BAM(256)
        self.bam3 = BAM(512)

        self.drop = nn.Dropout(p=drop)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x, get_bam=False):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        before_bam1 = x
        after_bam1 = self.bam1(before_bam1)
        x = self.pool(after_bam1)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        before_bam2 = x
        after_bam2 = self.bam2(before_bam2)
        x = self.pool(after_bam2)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        before_bam3 = x
        after_bam3 = self.bam3(before_bam3)
        x = self.pool(after_bam3)

        x = x.view(-1, 512 * (9 if self.crop == 48 else 4))
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))
        x = self.classifier(x)
        if get_bam:
            return x, [(before_bam1, after_bam1), (before_bam2, after_bam2), (before_bam3, after_bam3)]
        return x


if __name__ == '__main__':
    from torchsummary import summary

    crop = 40
    model = VggBAM(crop=crop).to('cuda')
    # model.eval()
    model(torch.zeros((1, 1, crop, crop)).to('cuda'))
    summary(model, (1, crop, crop))
