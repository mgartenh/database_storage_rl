USE TPCH;

SET GLOBAL local_infile=1;

DELETE FROM region;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/region.csv'
INTO TABLE region
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE region;

DELETE FROM nation;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/nation.csv'
INTO TABLE nation
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE nation;

DELETE FROM part;
LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/part.csv'
INTO TABLE part
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE part;

DELETE FROM supplier;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/supplier.csv'
INTO TABLE supplier
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE supplier;

DELETE FROM partsupp;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/partsupp.csv'
INTO TABLE partsupp
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE partsupp;

-- SELECT * FROM customer;
DELETE FROM customer;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/customer.csv'
INTO TABLE customer
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE customer;

DELETE FROM orders;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE orders;

DELETE FROM lineitem;

LOAD DATA LOCAL INFILE '/Users/mgartenhaus/CS598/data/lineitem.csv'
INTO TABLE lineitem
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n';

OPTIMIZE TABLE lineitem;
