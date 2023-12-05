import pandas as pd
import copy

def get_table_index_info_extremes(database):
    tables_list = pd.read_sql("SHOW TABLES", database)["Tables_in_TPCH"].tolist()
    index_table_mapping = dict()
    index_list = list()
    for table in tables_list:
        query_result = pd.read_sql(f"SHOW indexes FROM {table} WHERE key_name LIKE 'index_%'", database)
        index_table_mapping[table] = query_result["Column_name"].tolist()
        index_list += query_result["Key_name"].tolist()

    table_names = [x.split("_")[1] for x in index_list]

    table_index_info_noopt = dict()
    for i in range(len(index_list)):
        index = index_list[i]
        table = table_names[i]
        index_col = index.replace(f"index_{table}_", "")
        if table in table_index_info_noopt:
            table_index_info_noopt[table]["indexes"].append(index_col)
        else:
            table_index_info_noopt[table] = {
                "use_index_flag": False,
                "indexes": [index_col],
            }
    
    table_index_info_allopt = copy.deepcopy(table_index_info_noopt)

    for table in table_index_info_allopt:
        table_index_info_allopt[table]["use_index_flag"] = True

    return table_index_info_noopt, table_index_info_allopt

def get_table_index_info_inverse(table_index_info, table_index_info_noopt):
    table_index_info_inverse = dict()

    for table in table_index_info_noopt:
        table_index_info_inverse[table] = copy.deepcopy(table_index_info_noopt[table])
        if table in table_index_info:
            indexes = table_index_info[table]["indexes"]
            table_index_info_inverse[table]["indexes"] = list(set(table_index_info_inverse[table]["indexes"]) - set(indexes))
    
    return table_index_info_inverse
    