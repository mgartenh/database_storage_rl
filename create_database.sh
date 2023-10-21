total_gbs=.01
system=MAC

sh setup/01_docker_startup.sh
sh setup/02_setup_tpch.sh $total_gbs $system