"""
Load data and train model
"""
from model import CostNetwork
from cost import get_index_cost, get_query_cost

import random
from tqdm import tqdm
from time import sleep

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mysql.connector
import pandas as pd
import sqlparse
import sqlglot
from functools import partial
import json

dec_to_bin_dict = {}

#TODO: finish this and make sure it works
def get_cost(cursor, state, num_queries):
    def get_table_index_info(cursor, state):
        pass
    def sample_query():
        pass
    table_index_info = get_table_index_info(cursor, state)

    query_cost = 0
    for i in range(num_queries):
        query = sample_query()
        query_cost += get_query_cost(cursor, query, table_index_info)

    query_cost /= num_queries
    
    index_cost = 0
    
    #TODO: need to double check how to index cost is calculated
    index_cost = get_index_cost(cursor, table_index_info)
    
    #TODO: need to determine if change_cost will be added (maybe after midterm report)

    #TODO: determine hyperparamaters (coeffs of costs, alpha and beta??)
    total_cost = query_cost + index_cost
    
    return total_cost



#this is based off of the initial RL algorithm
#function will use model inference 
#with epsilon greedy method to determine next index state
def choose_action(model, num_indexes, eps=0.1):
    def dec_to_bin(num):
        assert num >= 0

        #if num in dec_to_bin_dict:
            #return dec_to_bin_dict[num]

        #return list of 1s and 0s based on integer num
        dec_num = num
        bin_num = []

        while dec_num >= 1:
            bin_num.append(dec_num % 2)
            dec_num = dec_num // 2
        
        while len(bin_num) < num_indexes:
            bin_num.insert(0, 0)

        #dec_to_bin_dict[num] = bin_num

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
                min_output = output
                action = i

    return dec_to_bin(action)

if __name__ == "__main__": 
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    #set hyperparameters for creating model
    num_indexes = 5 #TODO: get actual number of indexes
    num_hidden = 3
    hidden_dim = num_indexes

    model = CostNetwork(num_indexes, num_hidden, hidden_dim).to(device)

    input = torch.Tensor([0 for i in range(num_indexes)])
    output = None

    num_epochs = 10
    num_steps = 1000
    num_queries = 100

    learning_rate = 0.01

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) #TODO: check this and make sure it works


    database = mysql.connector.connect(
        user='root', 
        password='password',
        host='127.0.0.1', 
        port=3307,
        database="TPCH",
    )

    cursor = database.cursor()

    print("Beginning Training...")

    epochs_progress = tqdm(range(num_epochs), leave=True)

    for epoch in epochs_progress:
        steps_progress = tqdm(range(num_steps), leave=False)
    
        for step in steps_progress:
            optimizer.zero_grad()

            state = choose_action(model, num_indexes)
            input = torch.Tensor(state)
            output = model(input)

            #TODO:get target for labels using get_cost()
            # target = get_cost(cursor, state, num_queries)
            target = torch.Tensor([1])

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        steps_progress.close()
    epochs_progress.close()