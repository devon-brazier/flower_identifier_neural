from torch import nn

import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, hidden_units, input_features, drop_p=0.5):
        super().__init__()

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_features, 512)])
        
        # Add a variable number of more hidden layers
        for num in range(hidden_units):
            self.hidden_layers.extend(nn.Linear(512, 512))
        
        self.output = nn.Linear(512, 102)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):        
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)