SELECT "DROPPING TPCH DATABASE IF EXISTS";
DROP DATABASE IF EXISTS TPCH;

SELECT "CREATING TPCH DATABASE";
CREATE DATABASE TPCH;
USE TPCH;

CREATE TABLE region
(
    r_regionkey  INTEGER NOT NULL,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152),
    PRIMARY KEY(r_regionkey)
);

SELECT "CREATING TABLE nation";
CREATE TABLE nation
(
    n_nationkey  INTEGER NOT NULL,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INTEGER NOT NULL,
    n_comment    VARCHAR(152),
    PRIMARY KEY (n_nationkey),
    FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)

);

CREATE TABLE part
(
    p_partkey     BIGINT NOT NULL,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice DOUBLE PRECISION NOT NULL,
    p_comment     VARCHAR(23) NOT NULL,
    PRIMARY KEY(p_partkey)
);

CREATE TABLE supplier
(
    s_suppkey     BIGINT NOT NULL,
    s_name        CHAR(25) NOT NULL,
    s_address     VARCHAR(40) NOT NULL,
    s_nationkey   INTEGER NOT NULL,
    s_phone       CHAR(15) NOT NULL,
    s_acctbal     DOUBLE PRECISION NOT NULL,
    s_comment     VARCHAR(101) NOT NULL,
    PRIMARY KEY(s_suppkey),
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE partsupp
(
    ps_partkey     BIGINT NOT NULL,
    ps_suppkey     BIGINT NOT NULL,
    ps_availqty    BIGINT NOT NULL,
    ps_supplycost  DOUBLE PRECISION  NOT NULL,
    ps_comment     VARCHAR(199) NOT NULL,
    PRIMARY KEY(ps_partkey, ps_suppkey),
    FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
    FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
);

CREATE TABLE customer
(
    c_custkey     BIGINT NOT NULL,
    c_name        VARCHAR(25) NOT NULL,
    c_address     VARCHAR(40) NOT NULL,
    c_nationkey   INTEGER NOT NULL,
    c_phone       CHAR(15) NOT NULL,
    c_acctbal     DOUBLE PRECISION   NOT NULL,
    c_mktsegment  CHAR(10) NOT NULL,
    c_comment     VARCHAR(117) NOT NULL,
    PRIMARY KEY(c_custkey),
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

CREATE TABLE orders
(
    o_orderkey       BIGINT NOT NULL,
    o_custkey        BIGINT NOT NULL,
    o_orderstatus    CHAR(1) NOT NULL,
    o_totalprice     DOUBLE PRECISION NOT NULL,
    o_orderdate      DATE NOT NULL,
    o_orderpriority  CHAR(15) NOT NULL,  
    o_clerk          CHAR(15) NOT NULL, 
    o_shippriority   INTEGER NOT NULL,
    o_comment        VARCHAR(79) NOT NULL,
    PRIMARY KEY(o_orderkey),
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

CREATE TABLE lineitem
(
    l_orderkey    BIGINT NOT NULL,
    l_partkey     BIGINT NOT NULL,
    l_suppkey     BIGINT NOT NULL,
    l_linenumber  BIGINT NOT NULL,
    l_quantity    DOUBLE PRECISION NOT NULL,
    l_extendedprice  DOUBLE PRECISION NOT NULL,
    l_discount    DOUBLE PRECISION NOT NULL,
    l_tax         DOUBLE PRECISION NOT NULL,
    l_returnflag  CHAR(1) NOT NULL,
    l_linestatus  CHAR(1) NOT NULL,
    l_shipdate    DATE NOT NULL,
    l_commitdate  DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct CHAR(25) NOT NULL,
    l_shipmode     CHAR(10) NOT NULL,
    l_comment      VARCHAR(44) NOT NULL,
    PRIMARY KEY(l_orderkey, l_linenumber),
    FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
);
