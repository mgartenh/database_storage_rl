"""
Basic Neural Network Implementation
Input: prior state 
Output: recommended state 

Environment indexes based on recommended state and runs a set of sampled queries to determine loss.

Loss: Minimize Query Cost + Storage Cost + Index Change Cost
"""

# After thinking and discussing, ultimately realized that a single-step RL algorithm is the same as a neural network. 

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class CostNetwork(nn.Module):
    def __init__(self, num_indexes, num_hidden, hidden_dim):
        assert num_indexes > 0
        assert hidden_dim > 0
        assert num_hidden > 0

        super(CostNetwork, self).__init__()

        self.num_indexes = num_indexes #input and output dim
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim

        #self.flatten = nn.Flatten()

        layers = OrderedDict()
    
        layers['fc1'] = nn.Linear(in_features=num_indexes, out_features=hidden_dim)
        layers['relu1'] = nn.ReLU()

        for i in range(2, num_hidden):
            layers['fc' + str(i)] = nn.Linear(hidden_dim, hidden_dim)
            layers['relu' + str(i)] = nn.ReLU()

        layers['out'] = nn.Linear(hidden_dim, 1)

        self.neural_net = nn.Sequential(layers)

    def forward(self, x):
        # input: x is the prior state, shape is (1, num_indexes)
        # output: y is the recommended state, shape is (num_indexes) 
        #flatten = torch.flatten(x, 1)
        out = self.neural_net(x)

        return out