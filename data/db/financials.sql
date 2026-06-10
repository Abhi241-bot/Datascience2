-- Sample "company financials" database for the Analyst's Text-to-SQL tool.
-- Read-only at query time (the SQL guard blocks writes). Synthetic data.

DROP TABLE IF EXISTS segment_revenue;
DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS companies;

CREATE TABLE companies (
    company_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    sector       TEXT NOT NULL,
    hq_country   TEXT NOT NULL,
    founded_year INTEGER
);

CREATE TABLE financials (
    fin_id       INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(company_id),
    fiscal_year  INTEGER NOT NULL,
    revenue_musd     REAL,   -- total revenue, millions USD
    net_income_musd  REAL,   -- net income, millions USD
    rnd_musd         REAL,   -- R&D spend, millions USD
    employees        INTEGER
);

CREATE TABLE segment_revenue (
    seg_id       INTEGER PRIMARY KEY,
    company_id   INTEGER NOT NULL REFERENCES companies(company_id),
    fiscal_year  INTEGER NOT NULL,
    segment      TEXT NOT NULL,
    revenue_musd REAL
);

INSERT INTO companies VALUES
 (1,'Nimbus Cloud','NMBS','Technology','USA',2009),
 (2,'Helios Energy','HLOS','Energy','USA',1998),
 (3,'Veritas Pharma','VRTP','Healthcare','Switzerland',1985),
 (4,'Orchard Retail','ORCH','Consumer','USA',1972),
 (5,'Meridian Auto','MRDN','Industrials','Germany',1956);

INSERT INTO financials VALUES
 (1,1,2022, 8200, 1450, 1600, 21000),
 (2,1,2023,10250, 1980, 1950, 24500),
 (3,2,2022,15400, 2100,  320, 18000),
 (4,2,2023,14100, 1750,  340, 17600),
 (5,3,2022,12300, 3050, 2800, 26000),
 (6,3,2023,13450, 3320, 3050, 27200),
 (7,4,2022,22500, 1320,  120, 95000),
 (8,4,2023,23800, 1410,  130, 97000),
 (9,5,2022,31200, 2450, 1850, 88000),
 (10,5,2023,29800, 1980, 1900, 86500);

INSERT INTO segment_revenue VALUES
 (1,1,2023,'Cloud Platform', 6400),
 (2,1,2023,'Developer Tools',2350),
 (3,1,2023,'Other',          1500),
 (4,3,2023,'Oncology',       7200),
 (5,3,2023,'Cardiology',     4100),
 (6,3,2023,'Other',          2150),
 (7,5,2023,'Passenger Cars',21000),
 (8,5,2023,'Commercial',      6800),
 (9,5,2023,'Parts & Service', 2000);
