"""
Determine query and storage cost of queries

Code was written by Matthew Gartenhaus

"""
import sqlglot
import mysql.connector
from functools import partial
import json

def partial_transformer(node, table_index_info):
    if isinstance(node, sqlglot.exp.Table):
        table_name = node.this.output_name
        if table_name in table_index_info:
            use_index_flag = table_index_info[table_name]["use_index_flag"]
            indexes = [ f"index_{table_name}_{column}" for column in table_index_info[table_name]["indexes"]]
        else:
            use_index_flag = True
            indexes = list()

        table_hint = sqlglot.exp.IndexTableHint()
        table_hint.set("this", "USE" if use_index_flag else "IGNORE")
        indexes_identifier = sqlglot.exp.Identifier()
        indexes_identifier.set("this", ", ".join(indexes))
        table_hint.set("expressions", table_hint.expressions + [indexes_identifier])
        node.set("hints", node.expressions + [table_hint])
        return node
    return node

def get_query_cost(cursor, query, table_index_info) -> float:
    expression_tree = sqlglot.parse_one(query)
    transformer = partial(partial_transformer, table_index_info=table_index_info)
    transformed_tree = expression_tree.transform(transformer)
    index_specified_query = transformed_tree.sql()
    cursor.execute(f"EXPLAIN FORMAT='JSON' {index_specified_query}")

    query_cost = json.loads(cursor.fetchall()[0][0])["query_block"]["cost_info"]["query_cost"]

    return float(query_cost)

def get_index_cost(cursor, table_index_info) -> float:
    index_name_list = []
    for table_name in table_index_info:
        for column in table_index_info[table_name]["indexes"]:
            index_name_list.append(f"index_{table_name}_{column}")

    index_name_list_string = "('"+ "','".join(index_name_list) + "')"
        
    #print(index_name_list_string)

    cursor.execute(f"SELECT ROUND(SUM(stat_value * @@innodb_page_size / 1024 / 1024), 2) size_in_mb FROM mysql.innodb_index_stats WHERE stat_name = 'size' AND index_name != 'PRIMARY' AND database_name = 'TPCH' AND index_name IN {index_name_list_string}")
    return float(cursor.fetchone()[0])