import mysql.connector
import pandas as pd
import sqlparse
import sqlglot

database = mysql.connector.connect(
    user='root', 
    password='password',
    host='127.0.0.1', 
    port=3307,
    database="TPCH",
)

cursor = database.cursor()

statistics_queries = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE TABLE_SCHEMA = 'TPCH'
"""

keys_query = """
    SELECT 
        CASE WHEN constraint_name = "PRIMARY" THEN "PRIMARY" ELSE "FOREIGN" END AS key_type, 
        table_name, column_name, referenced_table_name, referenced_column_name
    FROM information_schema.key_column_usage 
    WHERE constraint_schema = 'TPCH' AND (constraint_name LIKE "%_ibfk_%" OR constraint_name = "PRIMARY")
"""
keys_data = pd.read_sql(keys_query, database)

col_data_query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE TABLE_SCHEMA = 'TPCH'
"""

table_column_info = (
    pd.read_sql(col_data_query, database)
)

schema_dictionary = dict()
for _, row in table_column_info.iterrows():
    table_name = row["TABLE_NAME"]
    if table_name not in schema_dictionary:
        schema_dictionary[table_name] = dict()
        schema_dictionary[table_name]["columns"] = dict()
    
    column_name = row["COLUMN_NAME"]

    if column_name not in schema_dictionary[table_name]:
        schema_dictionary[table_name]["columns"][column_name] = dict()
        
    schema_dictionary[table_name]["columns"][column_name]["data_type"] = str(row["DATA_TYPE"])

for table_name in schema_dictionary.keys():

    count_column_string = ""
    for col in schema_dictionary[table_name]["columns"].keys():
        count_column_string += f"COUNT(DISTINCT {col}) AS distinct_{col}, SUM(ISNULL({col})) AS null_{col}, "

    count_column_string = count_column_string[:-2]
    
    counts = pd.read_sql(f"SELECT COUNT(*) AS overall_count, {count_column_string} FROM {table_name}" , database).to_dict("records")[0]

    total_records = counts["overall_count"]

    for col in counts:
        if col == "overall_count":
            schema_dictionary[table_name]["count"] = int(counts[col])
        elif col.startswith("distinct_"):
            schema_dictionary[table_name]["columns"][col.replace("distinct_", "")]["distinct_count"] = int(counts[col])
            schema_dictionary[table_name]["columns"][col.replace("distinct_", "")]["distinct_percent"] = round(100 * counts[col] / total_records, 2)

        elif col.startswith("null_"):
            schema_dictionary[table_name]["columns"][col.replace("null_", "")]["null_count"] = int(counts[col])
            schema_dictionary[table_name]["columns"][col.replace("null_", "")]["null_percent"] = round(100 * counts[col] / total_records, 2)
        else:
            raise Exception()
        
index_count = 0
partition_count = 0
for table in schema_dictionary.keys():
    columns = schema_dictionary[table]["columns"]
    schema_dictionary[table]["index_candidates"] = list()
    schema_dictionary[table]["partition_candidates"] = list()
    for col in columns.keys():
        indexable = columns[col]["distinct_percent"] < 10
        columns[col]["indexable_flag"] = indexable
        partitionable = columns[col]["data_type"] in ["double", "int", "bigint"] and not indexable
        if indexable:
            schema_dictionary[table]["index_candidates"].append(col)
            index_count+=1
        elif partitionable:
            schema_dictionary[table]["partition_candidates"].append(col)
            partition_count+=1


    print(f"For {table}, {schema_dictionary[table]['index_candidates']} are index candidates")
    print(f"For {table}, {schema_dictionary[table]['partition_candidates']} are partition candidates")

print(f"There are {index_count} candidate index columns and {partition_count} candidate partition columns")


for table in schema_dictionary.keys():
    for index in schema_dictionary[table]["index_candidates"]:
        index_name = f"index_{table}_{index}"
        print(f"Creating {index_name}")
        cursor.execute(f"CREATE INDEX {index_name} ON {table} ({index})")