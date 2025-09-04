import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': '/media/yang/1305D8BDB8D46DEE/zzz/A2Net-main/resnet18-5c106cde.pth',
    'resnet34': '/media/WD_2T/ZYJ/duomotai_test/resnet34-333f7ec4 .pth',
    'resnet50': '/media/WD_2T/ZYJ/duomotai_test/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()

        self.channels = [64, 64 * block.expansion, 128 * block.expansion,
                         256 * block.expansion, 512 * block.expansion]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c0 = self.relu(x)
        c1 = self.maxpool(c0)

        c1 = self.layer1(c1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c0, c1, c2, c3, c4

    def forward(self, x):
        c0,c1,c2,c3,c4= self.base_forward(x)

        return c0,c1,c2,c3,c4


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("loading the {} pretrained weight from local .....".format(arch))
        state_dict = torch.load(model_urls[arch])
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


class Rsenet18(nn.Module):
    def __init__(self):
        super(Rsenet18, self).__init__()

        self.model = resnet18(pretrained=True)
        print("loading the pretrained resnet18 ......")
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16,kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=24,kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

    def forward(self,x):
        c0, c1, c2, c3, c4 = self.model(x)
        c0, c1, c2, c3, c4 = self.conv0(c0), self.conv1(c1), self.conv2(c2), self.conv3(c3), self.conv4(c4)
        return  c0, c1, c2, c3, c4

class Rsenet34(nn.Module):
    def __init__(self):
        super(Rsenet34, self).__init__()
        self.model = resnet34(pretrained=True)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

    def forward(self,x):
        c0, c1, c2, c3, c4 = self.model(x)
        c0, c1, c2, c3, c4 = self.conv0(c0), self.conv1(c1), self.conv2(c2), self.conv3(c3), self.conv4(c4)
        return c0, c1, c2, c3, c4

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = resnet50(pretrained=True)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16,kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=24,kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=96, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=320, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

    def forward(self,x):
        c0, c1, c2, c3, c4 = self.model(x)
        c0, c1, c2, c3, c4 = self.conv0(c0), self.conv1(c1), self.conv2(c2), self.conv3(c3), self.conv4(c4)
        return  c0, c1, c2, c3, c4

def resnet34(pretrained=False, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


def resnet152(pretrained=False, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


def resnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


def resnext101_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   replace_stride_with_dilation=[False, True, True], **kwargs)


if __name__ == '__main__':
    x = torch.randn((1,3,256,256))
    model = Rsenet18()
    c0, c1, c2, c3, c4 = model(x)
    print(1111)


    #resnet18   (b,64,128,128)  (b,64,64,64) (b,128,32,32) (b,256,32,32) (b,512,32,32)
    #resnet34   (b,64,128,128)  (b,64,64,64) (b,128,32,32) (b,256,32,32) (b,512,32,32)
    #resnet50   (b,64,128,128)  (b,256,64,64) (b,512,32,32) (b,1024,32,32) (b,2048,32,32)



