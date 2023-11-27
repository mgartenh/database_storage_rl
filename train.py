"""
Load data and train model
"""
from model import CostNetwork, choose_action
from cost import get_query_cost, get_index_cost
from index import get_table_index_info, get_table_index_info_extremes, get_table_index_info_inverse, get_indexes

import random
from tqdm import tqdm

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mysql.connector

import matplotlib.pyplot as plt

table_noopt = table_allopt = None

def get_cost(cursor, state, indexes, queries, num_queries, alpha=0.5, mode="train"):
    def sample_query(queries):
        return random.choice(queries)
    
    table = get_table_index_info(state, indexes)
    table_inverse = get_table_index_info_inverse(table, table_noopt)

    query_cost = 0
    
    for i in range(num_queries):
        query = sample_query(queries)

        if mode == "eval":
            query_cost += get_query_cost_actual(cursor, query, table_inverse) / get_query_cost_actual(cursor, query, table_noopt) 
        else:
            query_cost += get_query_cost(cursor, query, table_inverse) / get_query_cost(cursor, query, table_noopt) 
        
    query_cost /= num_queries
    
    index_cost = 0
    
    if max(state) > 0:
        index_cost = get_index_cost(cursor, table) / get_index_cost(cursor, table_allopt) 
        
    total_cost = alpha * query_cost + (1 - alpha) * index_cost

    return total_cost

def get_best_indexes(k, database, cursor, queries, num_queries):
    num_indexes, indexes, _ = get_indexes(database)

    index_costs = {}

    for state_idx in range(num_indexes):
        state_bin = [0 for i in range(num_indexes)]
        state_bin[state_idx] = 1

        index_costs[indexes[state_idx]] = get_cost(cursor, state_bin, indexes, queries, num_queries)

    index_costs = [(k, v) for k, v in index_costs.items()]
    index_costs.sort(key=lambda x: x[1])
    index_costs = [index_costs[i][0] for i in range(k)]

    return index_costs

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

    table_noopt, table_allopt = get_table_index_info_extremes(database)

    sql_reader = open("queries/test_queries.sql")
    queries = sql_reader.read().split(";")
    sql_reader.close()

    #indexes = ["index_lineitem_l_returnflag", "index_customer_c_nationkey", "index_lineitem_l_partkey", "index_lineitem_l_suppkey", "index_orders_o_orderstatus"]
    #num_indexes = len(indexes)
    num_queries = 10

    num_greedy_indexes = 5
    greedy_indexes = get_best_indexes(num_greedy_indexes, database, cursor, queries, num_queries)

    greedy_state = [1 for _ in range(len(greedy_indexes))]

    table_greedy_opt = get_table_index_info(greedy_state, greedy_indexes)

    print(f"Table Greedy Opt:\n {table_greedy_opt}\n")

    #determine actual num_indexes, indexes for each table
    #TODO: change this so it runs the neural network on every table before a final pass on combined resulting config

    num_indexes = 10
    indexes = get_best_indexes(num_indexes, database, cursor, queries, num_queries)

    num_hidden = 3
    hidden_dim = 10

    model = CostNetwork(num_indexes, num_hidden, hidden_dim).to(device)

    input = torch.Tensor([0 for i in range(num_indexes)])
    output = None

    num_epochs = 500
    num_queries = 25

    learning_rate = 1e-3

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("Initial Output:")

    with torch.no_grad():
        state = choose_action(model, num_indexes, eps=0)
        print(state)
        print([indexes[i] for i in range(num_indexes) if state[i] == 1])
        print(get_cost(cursor, state, indexes, queries, num_queries))

    curr_eps = start_eps = 0.5
    step_eps = 0.999
    end_eps = 0.1

    print("Beginning Training...")

    loss_values = []

    epochs_progress = tqdm(range(num_epochs), leave=True)

    for epoch in epochs_progress:
        optimizer.zero_grad()

        #print("Choosing Action...")
        state = choose_action(model, num_indexes, eps=0.5)
        input = torch.Tensor(state)

        #print("Running Model...")
        output = model(input)

        target = torch.Tensor([get_cost(cursor, state, indexes, queries, num_queries)])

        #print("Determining Loss")
        loss = criterion(output, target)

        loss_values.append(loss.item())

        #print("Backpropagation")
        loss.backward()
        optimizer.step()

        curr_eps = max(end_eps, curr_eps * step_eps)

    epochs_progress.close()


    print("Final Output:")

    with torch.no_grad():
        state = choose_action(model, num_indexes, eps=0)
        print(state)
        print([indexes[i] for i in range(num_indexes) if state[i] == 1])
        print(get_cost(cursor, state, indexes, queries, num_queries))

        table_ml_opt = get_table_index_info(state, indexes)

        print(f"Table ML Opt:\n {table_ml_opt}\n")

    plt.plot(range(num_epochs), loss_values)
    plt.show()