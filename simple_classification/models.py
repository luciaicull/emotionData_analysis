import torch as th
import torch.nn as nn

from .constants import FEATURE_KEYS

class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        
        self.input_size = input_size
        self.output_size = 5

        self.fc0 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size*3),
            nn.ReLU(),
            nn.Linear(self.input_size*3, self.input_size*5),
            nn.ReLU(),
            nn.Linear(self.input_size*5, self.input_size*3),
            nn.ReLU(),
            nn.Linear(self.input_size*3, self.input_size),
            nn.ReLU(),
            
            nn.Linear(self.input_size, self.output_size)
        )

    
    def forward(self, x):
        x = x.view(x.size(0)*x.size(1))
        x = self.fc0(x)

        return x
