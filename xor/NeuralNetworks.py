import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

class OneLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.model(x)
        return logits


class TwoLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.model(x)
        return logits