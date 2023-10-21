USE TPCH;

SET GLOBAL local_infile=1;

-- SELECT * FROM customer;
DELETE FROM customer;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/customer.csv'
INTO TABLE customer
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE customer;

DELETE FROM lineitem;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/lineitem.csv'
INTO TABLE lineitem
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE lineitem;

DELETE FROM nation;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/nation.csv'
INTO TABLE nation
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE nation;

DELETE FROM orders;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE orders;

DELETE FROM part;
LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/part.csv'
INTO TABLE part
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE part;

DELETE FROM partsupp;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/partsupp.csv'
INTO TABLE partsupp
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE partsupp;

DELETE FROM region;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/region.csv'
INTO TABLE region
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE region;

DELETE FROM supplier;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/supplier.csv'
INTO TABLE supplier
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE supplier;

-- select
-- 	l_returnflag,
-- 	l_linestatus,
-- 	sum(l_quantity) as sum_qty,
-- 	sum(l_extendedprice) as sum_base_price,
-- 	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
-- 	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
-- 	avg(l_quantity) as avg_qty,
-- 	avg(l_extendedprice) as avg_price,
-- 	avg(l_discount) as avg_disc,
-- 	count(*) as count_order
-- from
-- 	lineitem
-- where
-- 	l_shipdate <= date '1998-12-01'
-- group by
-- 	l_returnflag,
-- 	l_linestatus
-- order by
-- 	l_returnflag,
-- 	l_linestatus;