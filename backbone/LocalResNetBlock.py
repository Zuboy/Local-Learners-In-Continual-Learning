
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3( #stand3x3
    in_planes: int,
    out_planes: int,
    stride: int = 1,) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class LocalBasicBlock(nn.Module):
    expansion = 1

    def __init__( #constructor args
        self,
        in_planes: int,
        planes: int,
        num_classes: int,
        stride: int = 1,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        self.detach_input = detach_input #detach 

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.local_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.expansion * planes, num_classes), #own block pred
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.detach_input:
            x = x.detach()

        residual = self.shortcut(x) #grade flow con1,2, shortcut
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out)) 
        return F.relu(out + residual, inplace=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        local_logits = self.local_classifier(features)
        return features, local_logits #output adn local loss

    def local_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())


def local_block_step(
    block: LocalBasicBlock,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,) -> Tuple[torch.Tensor, float]:
    
    optimizer.zero_grad()
    features, local_logits = block(inputs)
    local_loss = criterion(local_logits, labels)
    local_loss.backward() #back prop inside block 
    optimizer.step()
    return features.detach(), local_loss.item() #return detach features and local loss
