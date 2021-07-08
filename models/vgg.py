import torch.nn as nn
import torch.nn.functional as F


class Vgg(nn.Module):
    def __init__(self, drop=0.4, attention=False, normalize_attn=True, num_classes=7):
        super().__init__()
        self.attention = attention

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv1b = nn.Conv2d(64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))

        self.conv2a = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.conv2b = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))

        self.conv3a = nn.Conv2d(512, 1024, (3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(1024, 2048, (3, 3), padding=(1, 1))

        self.conv4a = nn.Conv2d(2048, 4096, (3, 3), padding=(1, 1))
        self.conv4b = nn.Conv2d(4096, 4096, (3, 3), padding=(1, 1))

        # self.conv5a = nn.Conv2d(2048, 4096, (3, 3), padding=(1, 1))
        # self.conv5b = nn.Conv2d(4096, 4096, (3, 3), padding=(1, 1))
        #
        # self.conv6a = nn.Conv2d(4096, 4096, (3, 3), padding=(1, 1))
        # self.conv6b = nn.Conv2d(4096, 4096, (3, 3), padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(128)

        self.bn2a = nn.BatchNorm2d(256)
        self.bn2b = nn.BatchNorm2d(512)

        self.bn3a = nn.BatchNorm2d(1024)
        self.bn3b = nn.BatchNorm2d(2048)

        self.bn4a = nn.BatchNorm2d(4096)
        self.bn4b = nn.BatchNorm2d(4096)

        self.lin1 = nn.Linear(4096, 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.drop = nn.Dropout(p=drop)
        self.classify = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        l1 = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(l1)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = self.pool(x)
        l2 = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(l2)

        x = x.view(-1, 4096)
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))

        x = self.classify(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = Vgg().to('cuda')
    summary(model, (1, 48, 48))
