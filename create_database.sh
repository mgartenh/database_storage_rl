total_gbs=1
system=MAC

sh setup/01_docker_startup.sh
sh setup/02_setup_tpch.sh $total_gbs $system

source myenv/bin/activate
python3 setup/03_create_indexes.py