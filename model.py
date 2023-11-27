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

import random

from collections import OrderedDict

dec_to_bin_dict = {}

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
        layers['relu1'] = nn.LeakyReLU()

        for i in range(2, num_hidden):
            layers['fc' + str(i)] = nn.Linear(hidden_dim, hidden_dim)
            layers['relu' + str(i)] = nn.LeakyReLU()

        layers['out'] = nn.Linear(hidden_dim, 1)

        self.neural_net = nn.Sequential(layers)

    def forward(self, x):
        # input: x is the prior state, shape is (1, num_indexes)
        # output: y is the recommended state, shape is (num_indexes) 
        #flatten = torch.flatten(x, 1)
        out = self.neural_net(x)

        return out
    
#this is based off of the initial RL algorithm
#function will use model inference 
#with epsilon greedy method to determine next index state
def choose_action(model, num_indexes, eps=0.1):
    def dec_to_bin(num):
        assert num >= 0

        if num in dec_to_bin_dict:
            return dec_to_bin_dict[num]

        #return list of 1s and 0s based on integer num
        dec_num = num
        bin_num = []

        while dec_num >= 1:
            bin_num.append(dec_num % 2)
            dec_num = dec_num // 2
        
        while len(bin_num) < num_indexes:
            bin_num.insert(0, 0)

        dec_to_bin_dict[num] = bin_num

        return bin_num
    
    num_states = 2 ** num_indexes #each index has two states
    action = None
    
    p = random.random()

    if p < eps:
        #select random state
        action = random.randint(0, num_states - 1)
    else:
        #select best state based on model
        #this would be the action with the minimum estimated cost
        min_output = None

        for i in range(num_states):
            #print(dec_to_bin(i))
            input = torch.Tensor(dec_to_bin(i))
            #print(input)
            with torch.no_grad():
                output = model(input)

            if min_output == None or output < min_output:
                #print(output, min_output)
                min_output = output
                action = i

    return dec_to_bin(action)