import torch
import torch.nn as nn

filters = [64, 128, 256, 512]


class BottleNeck(nn.Module):
    expantion = 4

    def __init__(self, in_channel, out_channel, transition=None, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channel, self.expantion * out_channel, kernel_size=1, stride=1, bias=False)
        self.BN2 = nn.BatchNorm2d(self.expantion * out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.transition = transition

    def forward(self, x):
        y = self.conv1(x)
        y = self.BN1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.BN1(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.BN2(y)

        if self.transition is not None:
            x = self.transition(x)

        y = self.relu(x + y)
        return y


class Resnet(nn.Module):
    def __init__(self, block, num_classes=1000, layers=[3, 4, 6, 3], inchannels=3, **kwargs):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, filters[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.BN = nn.BatchNorm2d(filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.layer(block, filters[0], filters[0], layers[0], stride=1)
        self.stack2 = self.layer(block, filters[0], filters[1], layers[1], stride=2)
        self.stack3 = self.layer(block, filters[1], filters[2], layers[2], stride=2)
        self.stack4 = self.layer(block, filters[2], filters[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[3] * block.expantion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def layer(self, block, in_channels, out_channels, layer_numbers, stride):
        if out_channels != block.expantion * out_channels or stride != 1:
            if stride == 2:
                in_channels = in_channels * block.expantion
            else:
                in_channels = in_channels
            transition = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expantion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expantion)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, transition=transition))
        for _ in range(1, layer_numbers):
            layers.append(block(out_channels * block.expantion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = Resnet(BottleNeck, 7).to('cuda')
    model(torch.zeros((1, 1, 40, 40)).to('cuda'))
    summary(model, (1, 48, 48))
