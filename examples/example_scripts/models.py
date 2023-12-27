import torch
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim_2, output_dim)
        
    def forward(self, inp):
        x = inp["features"]
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)
    

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim_2, 1)
        
    def forward(self, inp):
        x = inp["features"]
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)
        