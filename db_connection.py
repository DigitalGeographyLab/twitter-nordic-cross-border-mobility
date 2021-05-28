import credentials
import io
from io import StringIO
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine, func, distinct


db_username = credentials.db_username
db_pwd = credentials.db_pwd

db_name = credentials.db_name
db_host = credentials.db_host
db_port = 5432

psyco_con = psycopg2.connect(database=db_name, user=db_username, password=db_pwd,
                       host=db_host)


engine_string = f"postgresql://{db_username}:{db_pwd}@{db_host}:{db_port}/{db_name}"

db_engine = create_engine(engine_string)

def read_sql_inmem_uncompressed(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(query=query, head="HEADER")
    conn = db_engine.raw_connection()
    cur = conn.cursor()
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    cur.close()
    return df

