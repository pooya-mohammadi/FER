import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ProjectorBlock, LinearAttentionBlock


class Vgg(nn.Module):
    def __init__(self, drop=0.2, attention=True, normalize_attn=True, num_classes=7):
        super().__init__()
        self.attention = attention
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5a = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.bn5a = nn.BatchNorm2d(512)
        self.bn5b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512, 512)

        self.drop = nn.Dropout(p=drop)
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        if self.attention:
            self.lin2 = nn.Linear(512 * 3, 4096)
            self.classify = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)

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
        l2 = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(l2)
        # print(x.shape)

        x = F.relu(self.bn5a(self.conv5a(x)))
        l3 = F.relu(self.bn5b(self.conv5b(x)))
        x = self.pool(l3)

        x = x.view(-1, 512)
        g = F.relu(self.drop(self.lin1(x)))
        # g = F.relu(self.drop(self.lin2(x)))

        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizexC
            # classification layer
            x = F.relu(self.drop(self.lin2(g)))
            x = self.classify(x)  # batch_sizexnum_classes
        else:
            x = self.classify(g)
        return x

# class Vgg(VggFeatures):
#     def __init__(self, drop=0.2):
#         super().__init__(drop)
#         self.lin3 = nn.Linear(4096, 7)
#
#     def forward(self, x):
#         x = super().forward(x)
#         x = self.lin3(x)
#         return x
