## Installation Instructions
1. Create a virtual environment for python 3.8 via the following command: `python3.8 -m venv myenv`
2. Activate the virtual environment with `source myenv/bin/activate`
3. Open create_database.sh and update total_gbs to number of GBs of TPCH data you want and your system. Most common options are MAC, LINUX, or WINDOWS
4. Run `sh create_database.sh` to create a docker container running a mysql database with your specifications. This may require a little debugging based on your system setup.
5. Once that is complete, you can test out connecting to the database through a jupyter notebook.