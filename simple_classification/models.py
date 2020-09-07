import torch as th
import torch.nn as nn

from .constants import FEATURE_KEYS

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        '''
        self.input_size = param['input']
        self.output_size = param['output']
        '''
        self.input_size = len(FEATURE_KEYS)
        self.output_size = 5

        self.fc0 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size*3),
            
            nn.Linear(self.input_size*3, self.input_size*2),
            nn.Linear(self.input_size*2, self.input_size),
            nn.ReLU(),
            
            nn.Linear(self.input_size, self.output_size)
        )
        '''
        self.fc1 = nn.Sequential(
            nn.Linear()
        )
        '''

    
    def forward(self, x):
        x = x.view(x.size(0)*x.size(1))
        x = self.fc0(x)

        return x
