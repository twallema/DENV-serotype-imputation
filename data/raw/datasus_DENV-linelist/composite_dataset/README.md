## Origin of these data

### 1996-1999

Updated data received by PI Bento in 2025.

### 2000-2007

Downloaded from [datasus](https://datasus.saude.gov.br/transferencia-de-arquivos/#) by Siyu Chen on May 21, 2025.

### 2008

Data received by PI Bento in 2021.

### 2009-2025

Downloaded from [datasus](https://datasus.saude.gov.br/transferencia-de-arquivos/#) by Siyu Chen on May 21, 2025.

## Alterations made to these data in Excel

+ `composite_1997.csv`: Line 64271 was removed because of an invalid date "2567-02-06".

+ `composite_2001.csv`:

    + On line 144810, column "DT_SIN_PRI" an invalid date "201-05-08" was rectified to "2001-05-08".
    + On line 377540, column "DT_SIN_PRI" an invalid date "1200-11-14" was rectified to "2001-11-14".
    + On line 2954, column "DT_FEBRE" an invalid date "1-02-01" was rectified to "2001-02-01".
    + Line 27603 was removed because of an invalid date "820-11-11".
    + On line 379632, column "DT_FEBRE" an invalid date "1200-02-06" was rectified to "2001-02-06".
    + On line 382107, column "DT_FEBRE" an invalid date "200-01-22" was rectified to "2001-01-22".
    + On line 385438, column "DT_FEBRE" an invalid date "1001-12-25" was rectified to "2001-12-25".
    + On line 457453, column "DT_FEBRE" an invalid date "1-05-22" was rectified to "2001-05-22".

+ `composite_2002.csv`: In this dataset, there were roughly 50 similar mistakes, about 10 in the column "DT_SIN_PRI" and about 40 in the column "DT_FEBRE". Because fixing these mistakes takes about 20 minutes for each year, I've abstained from manual rectification of dates from 2002 onwards.

+ `composite_2008.csv`: In this dataset, line 617303 was removed because the municipality ID was '33p440'.