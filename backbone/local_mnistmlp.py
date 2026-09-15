# Copy of local Mnistmlp but with local heads. 16.05.26

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from backbone import MammothBackbone, num_flat_features, register_backbone, xavier


class LocalMNISTMLP(MammothBackbone):
    def __init__(self, input_size=28*28, hidden_size=100, num_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.local_head1 = nn.Linear(hidden_size, num_classes)
        self.local_head2 = nn.Linear(hidden_size, num_classes)

        self.blocks = [self.fc1, self.fc2]
        self.heads = [self.local_head1, self.local_head2]


##########################################################################
# self learning , local heads
    def forward(self, x, returnt='out'):
        
        h = x.view(-1, num_flat_features(x))
        for block in self.blocks:
            h = F.relu(block(h))
        out = self.heads[-1](h)

        if returnt == 'out':
            return out
        elif returnt == 'features':
            return h
        elif returnt == 'full':
            return out, h

        raise NotImplementedError("Unknown return type")

    def local_forward(self, x):
        h = x.view(-1, num_flat_features(x))
        outs = []
        for block, head in zip(self.blocks, self.heads):
            h = F.relu(block(h))
            outs.append(head(h))
            h = h.detach()          # decouple the next block from this one
        return outs

    def forward_all_heads(self, x):
        return self.local_forward(x)

######################################################


@register_backbone("local-mnistmlp")
def local_mnistmlp(num_classes: int, mlp_hidden_size: int = 100):
    return LocalMNISTMLP(
        input_size=28 * 28,
        hidden_size=mlp_hidden_size,
        num_classes=num_classes
    )
