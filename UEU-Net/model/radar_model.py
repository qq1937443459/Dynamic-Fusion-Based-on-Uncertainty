import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed import get_2d_sincos_pos_embed
from param import args
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import timm.models.vision_transformer
matplotlib.use('TKAgg')
from models2 import models_vit
from torchvision import models


# class radar_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.r_model = models_vit.__dict__['vit_base_patch16'](
#             num_classes=10,
#             drop_path_rate=0.5,
#             global_pool=False,
#             )
#         self.load_model()
#     def load_model(self):
#         path = "D:\ywhgesture\RGB_radar\models2\st_gcn.kinetics.pt"
#         self.r_model.load_state_dict(torch.load(path),strict=False)
#     def forward(self,radar):
#         feat_r = self.r_model(radar)
#         return feat_r

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

class ResNet10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet10, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet10(num_classes=10):
    return ResNet10(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


if __name__=="__main__":
    # 加载ResNet-18的预训练模型
    pretrained_resnet18 = models.resnet101(pretrained=True)

    # 定义轻量级ResNet-10
    model = resnet10(num_classes=10)
    # 初始化ResNet-10的权重，使用预训练的ResNet-18的权重
    def load_pretrained_weights(model, pretrained_model):
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    load_pretrained_weights(model, pretrained_resnet18)
    input = torch.randn(8, 3, 408, 524)
    out = model(input)