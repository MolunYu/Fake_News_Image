import torch
import torch.nn as nn
from FakeNewsDataset import FakeNewsDataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transform
from resnet_copy import resnet50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
num_classes = 2
batch_size = 64
learning_rate = 0.001
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

train_dataset = FakeNewsDataset(train=True, transform=transform)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class FFTNet(nn.Module):
    def __init__(self, num_class=2) -> None:
        super().__init__()

        self.conv1 = conv3x3(1, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # out = self.fc(out)

        return out


class FakeNewsModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained=True)
        self.fft = FFTNet()
        self.fc = nn.Linear(2048 + 128, 2)

    def forward(self, *args):
        semantic_feature = self.resnet(args[0])
        physical_feature = self.fft(args[1])
        feature = torch.cat((semantic_feature, physical_feature), 1)
        out = self.fc(feature)

        return out
