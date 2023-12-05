## Installation Instructions
1. Create a virtual environment for python 3.8 via the following command: `python3.8 -m venv myenv`
2. Activate the virtual environment with `source myenv/bin/activate`
3. Open create_database.sh and update total_gbs to number of GBs of TPCH data you want and your system. Most common options are MAC, LINUX, or WINDOWS
4. Run `sh create_database.sh` to create a docker container running a mysql database with your specifications. This may require a little debugging based on your system setup.
5. Once that is complete, you can test out connecting to the database through a jupyter notebook. It is recommended to try out the `baselines.ipynb` notebook.
6. For training of greedyopt and mlopt, examine the train.py file and update test_query_13 to be `False`. This will disable the query skew instruction.
7. Within the virtual environment, run from the command line `python3 train.py`. It will output a greedyopt table_index_info at the start and will output an mlopt table_index_info at the end. The printed out dictionaries can be copied over to baselines.ipynb and can be used to update `table_index_info_greedyopt` and `table_index_info_mlopt` variables, respectively.
8. Within the `baselines.ipynb notebook`, under the section titled `Define Queries DBA Can Look At`, this can be used to examine a subset of queries to define HumOpt and update `table_index_info_humopt` accordingly. `table_index_info_noopt` and `table_index_info_allopt` do not need to be updated.
9. Update the `test_query_13` variable in `baselines.ipynb` to be False and run the notebook to get performance results.
10. The final results can then be exported to excel for evaluation and comparison.
11. If wanting to do query skew experiments for query 13, update `train.py` so that test_query_13 is True and update total_query_13 to be 5 (25% skew), 10 (50% skew), 15 (75% skew), and 20 (100% skew).
12. Within the virtual environment, run from the command line `python3 train.py`. Leverage the mlopt output to update the appropriate skew variable in baselines.ipynb (`table_index_info_mlopt_25`, `table_index_info_mlopt_50`, `table_index_info_mlopt_75`, `table_index_info_mlopt_100`)
13. Within the `baselines.ipynb` notebook, update the appropriate `table_index_info_mlopt_SKEW` variable and update `test_query_13` variable to True. Execute the notebook.