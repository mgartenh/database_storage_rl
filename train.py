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

import matplotlib.pyplot as plt

dec_to_bin_dict = {}

def get_indexes(database):
    tables_list = pd.read_sql("SHOW TABLES", database)["Tables_in_TPCH"].tolist()
    index_table_mapping = dict()
    index_list = list()
    for table in tables_list:
        query_result = pd.read_sql(f"SHOW indexes FROM {table} WHERE key_name LIKE 'index_%'", database)
        index_table_mapping[table] = query_result["Column_name"].tolist()
        index_list += query_result["Key_name"].tolist()
    
    return len(index_list), index_list

#TODO: finish this and make sure it works
def get_cost(cursor, state, indexes, queries, num_queries):
    def get_table_index_info(state, indexes):
        table_index_info = dict()

        table_names = [x.split("_")[1] for x in indexes]
        #print(table_names)

        for i in range(len(indexes)):
            if state[i] == 1:
                index = indexes[i]
                table = table_names[i]
                index_col = index.replace(f"index_{table}_", "")
                #print(table)
                #print(index_col)
                if table in table_index_info:
                    table_index_info[table]["indexes"].append(index_col)
                else:
                    table_index_info[table] = {
                        "use_index_flag": True,
                        "indexes": [index_col],
                    }
        #print(table_index_info)
        return table_index_info
    def sample_query(queries):
        #return queries[0]
        #n = random.randint(0, len(queries)-1)
        #print(n)
        #return queries[n]
        return random.choice(queries)
    table_index_info = get_table_index_info(state, indexes)
    #print(table_index_info)
    query_cost = 0
    for i in range(num_queries):
        #query = sample_query(queries)
        query = queries[i]
        #print(query)
        query_cost += get_query_cost(cursor, query, table_index_info)

    query_cost /= num_queries
    
    index_cost = 0
    
    #TODO: need to double check how to index cost is calculated
    if sum(state) > 0:
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

if __name__ == "__main__": 
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"


    database = mysql.connector.connect(
        user='root', 
        password='password',
        host='127.0.0.1', 
        port=3307,
        database="TPCH",
    )

    cursor = database.cursor()

    sql_reader = open("queries/test_queries.sql")
    queries = sql_reader.read().split(";")
    sql_reader.close()

    #set hyperparameters for creating model
    #num_indexes, indexes = get_indexes(database) #TODO: get actual number of indexes

    #TODO: change this
    #num_indexes = 5
    #indexes = random.choices(indexes, k=num_indexes)
    indexes = ["index_lineitem_l_returnflag", "index_customer_c_nationkey", "index_lineitem_l_partkey", "index_lineitem_l_suppkey", "index_orders_o_orderstatus"]
    num_indexes = len(indexes)
    
    num_hidden = 3
    hidden_dim = 10

    model = CostNetwork(num_indexes, num_hidden, hidden_dim).to(device)

    input = torch.Tensor([0 for i in range(num_indexes)])
    output = None

    num_epochs = 500
    num_steps = 1
    num_queries = 10

    learning_rate = 0.1

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) #TODO: check this and make sure it works

    print("Initial Output:")

    with torch.no_grad():
        state = choose_action(model, num_indexes, eps=0)
        print(state)
        print([indexes[i] for i in range(num_indexes) if state[i] == 1])
        print(get_cost(cursor, state, indexes, queries, num_queries))

    print("Beginning Training...")

    loss_values = []

    epochs_progress = tqdm(range(num_epochs), leave=True)

    for epoch in epochs_progress:
        steps_progress = tqdm(range(num_steps), leave=False)
    
        for step in steps_progress:
            optimizer.zero_grad()

            #print("Choosing Action...")
            state = choose_action(model, num_indexes)
            input = torch.Tensor(state)

            #print("Running Model...")
            output = model(input)

            #TODO:get target for labels using get_cost()
            target = torch.Tensor([get_cost(cursor, state, indexes, queries, num_queries)])
            #target = torch.Tensor([1])

            #print("Determining Loss")
            loss = criterion(output, target)

            loss_values.append(loss.item())

            #print("Backpropagation")
            loss.backward()
            optimizer.step()

        steps_progress.close()
    epochs_progress.close()


    print("Final Output:")

    with torch.no_grad():
        state = choose_action(model, num_indexes, eps=0)
        print(state)
        print([indexes[i] for i in range(num_indexes) if state[i] == 1])
        print(get_cost(cursor, state, indexes, queries, num_queries))

    plt.plot(range(num_epochs), loss_values)
    plt.show()