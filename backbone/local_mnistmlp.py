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

        # self.classifier = nn.Linear(hidden_size, num_classes)


##########################################################################
#self learning , local heads
    def forward(self, x, returnt='out'):
        x = x.view(-1, num_flat_features(x))
        h1 = F.relu(self.fc1(x))                    # L1
        h2 = F.relu(self.fc2(h1))                   # L2
        out = self.local_head2(h2)

        if returnt == 'out':
            return out
        elif returnt == 'features':
            return h2
        elif returnt == 'full':
            return out, h2

        raise NotImplementedError("Unknown return type")

    def local_forward(self, x):
        x = x.view(-1, num_flat_features(x))

        h1 = F.relu(self.fc1(x))
        out1 = self.local_head1(h1)    
        h2 = F.relu(self.fc2(h1.detach()))  #only L2 
        out2 = self.local_head2(h2)

        return out1, out2
    
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
