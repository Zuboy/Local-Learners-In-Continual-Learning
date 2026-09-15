
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import relu
from backbone import MammothBackbone, register_backbone
from backbone.ResNetBlock import BasicBlock, conv3x3


class LocalResNet(MammothBackbone):
    def __init__(self, num_classes: int, nf: int = 64) -> None:
        super().__init__()
        self.in_planes = nf

        stem = nn.Sequential(
            conv3x3(3, nf),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
        )
        layer1 = self._make_layer(nf, 2, stride=1)
        layer2 = self._make_layer(nf * 2, 2, stride=2)
        self.fc1 = nn.Sequential(stem, layer1, layer2)

        layer3 = self._make_layer(nf * 4, 2, stride=2)
        layer4 = self._make_layer(nf * 8, 2, stride=2)
        self.fc2 = nn.Sequential(layer3, layer4)

        self.local_head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 2, num_classes),
        )
        self.local_head2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, num_classes),
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def local_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.fc1(x)
        out1 = self.local_head1(h1)

        h2 = self.fc2(h1.detach().to(x.device)) #for faster
        out2 = self.local_head2(h2)
        return out1, out2

    def forward_all_heads(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.local_forward(x)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        feature = h2.mean(dim=(2, 3))

        if returnt == 'features':
            return feature
        out = self.local_head2(h2)
        if returnt == 'out':
            return out
        elif returnt == 'full':
            return out, feature

@register_backbone("local-resnet18")
def local_resnet18(num_classes: int, num_filters: int = 64) -> LocalResNet:
    return LocalResNet(num_classes, num_filters)
