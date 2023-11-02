total_gbs=$1
system=$2

rm -rf tpch-dbgen
git clone https://github.com/electrum/tpch-dbgen.git
cd tpch-dbgen/
cp makefile.suite makefile
# update variables
sed -i'' -e 's/CC      =/CC = gcc/g' makefile
sed -i'' -e 's/DATABASE=/DATABASE = ORACLE/g' makefile
sed -i'' -e "s/MACHINE =/MACHINE = $system/g" makefile
sed -i'' -e 's/WORKLOAD =/WORKLOAD = TPCH/g' makefile
make

cd ..
rm -rf data
mkdir data
cd data
cp ../tpch-dbgen/dbgen .
cp ../tpch-dbgen/dists.dss .
./dbgen -s $total_gbs
for i in `ls *.tbl`; do sed 's/.$//' $i > $(echo $i | sed 's/.tbl/.csv/g'); done;
rm -r *.tbl

cd ..
mysql --host=127.0.0.1 --user=root --password="password" --port=3307 < setup/import_data.sql