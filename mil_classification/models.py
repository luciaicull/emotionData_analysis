import torch as th
import torch.nn as nn

class AdaptivePoolingClassifier(nn.Module):
    def __init__(self, device, input_size):
        super(AdaptivePoolingClassifier, self).__init__()
        self.device = device

        self.input_size = input_size
        self.output_size = 5
        
        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, self.input_size*3),
            #nn.ReLU(),

            nn.Linear(self.input_size*3, self.input_size*2),
            #nn.ReLU(),

            nn.Linear(self.input_size*2, self.input_size),
            nn.ReLU(),


            nn.Linear(self.input_size, self.output_size),
            #nn.ReLU()
        )

        self.alpha = nn.Parameter(th.ones(self.output_size))


    def forward(self, x):
        predicted_x = th.zeros(x.shape[0], x.shape[1], self.output_size).to(self.device)
        for i, row in enumerate(x[0]):
            predicted_row = self.predictor(row)
            predicted_x[0, i, :] = predicted_row
        
        # pooling
        alpha_x = self.alpha * predicted_x
        max_values = th.max(alpha_x, axis=1, keepdim=True).values
        soften_max = th.exp(alpha_x - max_values)
        weight = soften_max / th.sum(soften_max, axis=1, keepdim=True)
        result = th.sum(predicted_x * weight, axis=1, keepdim=False)

        return result
