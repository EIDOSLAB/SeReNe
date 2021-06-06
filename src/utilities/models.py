import torch.nn.functional as F
from torch import nn

"""
LENET 5
"""


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=True)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, bias=True)
        self.r2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10, bias=True)

    def forward(self, img):
        output = self.conv1(img)
        output = self.r1(output)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = self.r2(output)
        output = F.max_pool2d(output, 2)
        output = output.view(img.size(0), -1)
        output = self.fc1(output)
        output = self.r3(output)
        output = self.fc2(output)

        return output


"""
LENET 300
"""


class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300, bias=True)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


"""
RESNET CIFAR10
"""


class ResnetLambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(ResnetLambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Increses dimension via padding, performs identity operations
                """
                self.shortcut = ResnetLambdaLayer(lambda x:
                                                  F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                        "constant",
                                                        0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option="A"):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(64, num_classes)

    # self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        pool_size = int(out.size(3))
        out = F.avg_pool2d(out, pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(option="A"):
    return ResNet(ResnetBlock, [3, 3, 3], option=option)


def resnet32(option="A"):
    return ResNet(ResnetBlock, [5, 5, 5], option=option)


def resnet44(option="A"):
    return ResNet(ResnetBlock, [7, 7, 7], option=option)


def resnet56(option="A"):
    return ResNet(ResnetBlock, [9, 9, 9], option=option)


def resnet110(option="A"):
    return ResNet(ResnetBlock, [18, 18, 18], option=option)


def resnet1202(option="A"):
    return ResNet(ResnetBlock, [200, 200, 200], option=option)


"""
VGG CIFAR10 1
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG1L(nn.Module):
    def __init__(self, vgg_name="VGG16"):
        super(VGG1L, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


"""
VGG CIFAR10 2
"""


class VGG2L(nn.Module):
    def __init__(self, classes=10):
        super(VGG2L, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers():
        layers = []
        layers += [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(64, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.3)]

        layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(64, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(128, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(128, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]

        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        return nn.Sequential(*layers)


"""
ALEXNET CIFAR100
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
