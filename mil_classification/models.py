import torch as th
import torch.nn as nn

class AdaptivePoolingClassifier(nn.Module):
    def __init__(self, device, feature_num, pooling):
        super(AdaptivePoolingClassifier, self).__init__()
        self.device = device
        self.pooling = pooling

        self.input_size = feature_num
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
        
        # auto pooling
        if self.pooling == "auto":
            result = self.auto_pooling(predicted_x)
        # max pooling
        elif self.pooling == "max":
            result = self.max_pooling(predicted_x)
        # mean pooling
        elif self.pooling == "mean":
            result = self.mean_pooling(predicted_x)
        # soft max pooling
        elif self.pooling == "softmax":
            result = self.soft_max_pooling(predicted_x)

        return result

    def auto_pooling(self, predicted_x):
        alpha_x = self.alpha * predicted_x
        max_values = th.max(alpha_x, axis=1, keepdim=True).values
        soften_max = th.exp(alpha_x - max_values)
        weight = soften_max / th.sum(soften_max, axis=1, keepdim=True)
        result = th.sum(predicted_x * weight, axis=1, keepdim=False)

        return result

    def max_pooling(self, predicted_x):
        max_values = th.max(predicted_x, axis=1).values
        return max_values
        

    def mean_pooling(self, predicted_x):
        mean_values = th.mean(predicted_x, axis=1)
        return mean_values

    def soft_max_pooling(self, predicted_x, alpha=1):
        max_values = th.max(predicted_x, axis=1, keepdim=True).values
        soften_max = th.exp(alpha*(predicted_x - max_values))
        weight = soften_max / th.sum(soften_max, axis=1, keepdim=True)
        result = th.sum(predicted_x * weight, axis=1, keepdim=False)

        return result
