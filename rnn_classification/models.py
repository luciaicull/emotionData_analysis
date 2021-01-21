import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, device, feature_num, hidden_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.device = device

        self.input_size = feature_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 5

        self.lstm_layer = nn.LSTM(input_size=self.input_size, 
                                  hidden_size=self.hidden_size, 
                                  num_layers=self.num_layers,
                                  batch_first=True,
                                  bidirectional=True
                                  )

        self.softmax_layer = nn.Softmax(dim=1)

        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        # lstm
        out, (h_n, c_n) = self.lstm_layer(x)
        hidden = h_n.view(-1, self.hidden_size*2, self.num_layers)
        last_hidden = hidden[:, :, 0].unsqueeze(dim=2)

        # get attention weight
        attn_weights = th.bmm(out, last_hidden).squeeze(dim=2)
        soft_attn_weights = self.softmax_layer(attn_weights)
        
        # context
        context = th.bmm(out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        output = self.last_layer(context)

        return output
