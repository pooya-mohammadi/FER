import torch
from torch.nn import functional as F
from torch import nn
from models.cbam import CBAM


class VggCBAM(nn.Module):
    def __init__(self, drop=0.2, crop=40, **kwargs):
        super().__init__()
        self.cbam_blocks = kwargs['cbam_blocks']
        self.residual_cbam = kwargs['residual_cbam']
        self.crop = crop
        self.linear_size = 1024
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5a = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5b = nn.Conv2d(1024, 1024, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.bn5a = nn.BatchNorm2d(1024)
        self.bn5b = nn.BatchNorm2d(1024)

        self.lin1 = nn.Linear(int(self.linear_size * self.crop / 20), 4096)
        # self.lin1 = nn.Linear(int(self.linear_size), 4096)

        self.lin2 = nn.Linear(4096, 4096)

        self.cbam0 = CBAM(64, self.residual_cbam)
        self.cbam1 = CBAM(128, self.residual_cbam)
        self.cbam2 = CBAM(256, self.residual_cbam)
        self.cbam3 = CBAM(512, self.residual_cbam)
        self.cbam4 = CBAM(1024, self.residual_cbam)

        self.drop = nn.Dropout(p=drop)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x, get_cbam=False):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        if 0 in self.cbam_blocks:
            before_bam0 = x
            after_bam0 = self.cbam0(before_bam0)
            x = self.pool(after_bam0)
        else:
            x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        if 1 in self.cbam_blocks:
            before_bam1 = x
            after_bam1 = self.cbam1(before_bam1)
            x = self.pool(after_bam1)
        else:
            x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        if 2 in self.cbam_blocks:
            before_bam2 = x
            after_bam2 = self.cbam2(before_bam2)
            x = self.pool(after_bam2)
        else:
            x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        if 3 in self.cbam_blocks:
            before_bam3 = x
            after_bam3 = self.cbam3(before_bam3)
            x = self.pool(after_bam3)
        else:
            x = self.pool(x)

        x = F.relu(self.bn5a(self.conv5a(x)))
        x = F.relu(self.bn5b(self.conv5b(x)))
        if 4 in self.cbam_blocks:
            before_bam4 = x
            after_bam4 = self.cbam4(before_bam4)
            x = self.pool(after_bam4)
        else:
            x = self.pool(x)

        x = x.view(-1, int(self.linear_size * self.crop / 20))
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))
        x = self.classifier(x)
        if get_cbam:
            return x, [
                (before_bam1, after_bam1),
                (before_bam2, after_bam2),
                (before_bam3, after_bam3),
                (before_bam4, after_bam4)
            ]
        return x


if __name__ == '__main__':
    from torchsummary import summary

    crop = 80
    model = VggCBAM(crop=80, cbam_blocks=(0, 1, 2, 3, 4), residual_cbam=True)
    # model.eval()
    model(torch.zeros((1, 1, crop, crop)).to('cuda'))
    summary(model, (1, crop, crop))
