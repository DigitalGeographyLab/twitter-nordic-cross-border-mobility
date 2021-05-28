import csv
from datetime import datetime
import db_connection as db_con
import geojson
import geopandas as gpd
import io
from io import StringIO
import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd
import pickle
import psycopg2
import psycopg2.extras as extras
from pyproj import CRS
from shapely.geometry import Point, LineString, Polygon
import sqlalchemy.pool as pool
from sqlalchemy import create_engine, func, distinct
import sys
import tempfile

print('Start')

script_start = datetime.now()
def execute_values(conn, df, table):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    Change to lat2, lon2 etc if using luxemburg tables
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "UPDATE %s AS t SET post_region = data.post_region FROM (VALUES %%s) AS data (row_id,post_region) WHERE t.row_id = data.row_id" % (table)
    cursor = conn.cursor()
    try:
        print('Uploading')
        extras.execute_values(cursor, query, tuples, page_size=1000)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()
def parallelize_dataframe(df, func, n_cores=2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def getconn():
    c = psycopg2.connect(user=db_con.db_username, host=db_con.db_host, dbname=db_con.db_name, password=db_con.db_pwd)
    return c

mypool = pool.QueuePool(getconn, max_overflow=10, pool_size=20)

def multi_read_sql_inmem_uncompressed(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(query=query, head="HEADER")
    conn = db_engine
    cur = conn.cursor()
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    cur.close()
    db_engine.close()
    return df

# Select table to update
table = 'pre_covid'
# Create connection
conn = db_con.db_engine.raw_connection()
# Initialize cursor
cur = conn.cursor()
# SQL statement
findCountSql = f"SELECT row_id FROM {table} ORDER BY row_id DESC LIMIT 1"
# Execute sql query
findCount = db_con.read_sql_inmem_uncompressed(findCountSql, db_con.db_engine)
# Find the last tweet
last_tweet = findCount['row_id'][0]
# Close connection
cur.close()
conn.close()
print(f"{last_tweet} number of records")

print('Read in NUTS.')
nuts = gpd.read_file(r"NUTS2.gpkg")

# Update parameters
batch_size = 80000

conn_psyco = db_con.psyco_con

def getClosestRegion(point):
    proj4_txt = f'+proj=eqc +lat_0={point.y} +lon_0={point.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    nuts_d = nuts.to_crs(proj4_txt)
    polygon_index = nuts_d.distance(point).sort_values().index[0]
    return nuts_d['NUTS_ID'].loc[polygon_index]

def get_region(point):
    point = Point(point[0],point[1])
    set_region = ''
    for idx, reg in nuts.iterrows():
        if point.within(reg['geometry']):
            set_region = reg['NUTS_ID']
        else:
            pass
    if set_region == '':
        return getClosestRegion(point)
    else:
        return set_region

def postRegionAssign(start_number):
    starttime = datetime.now()
    max_number = start_number+batch_size
    query = f'SELECT * FROM {table} WHERE row_id <{max_number} AND row_id>={start_number} ORDER BY row_id ASC LIMIT {batch_size}'
    conn = mypool.connect()
    data = multi_read_sql_inmem_uncompressed(query, conn)
    conn.close()
    print(f'Data read, rows: {start_number} - {max_number}')
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']))
    print('GeoDataFrame created.')
    gdf.crs = "EPSG:4326"
    gdf = gdf.to_crs(nuts.crs)
    del data
    print('DataFrame deleted.')
    withinRegion = gpd.sjoin(gdf,nuts, how='inner', op='within')
    print('Spatial join done.')
    withinRegion = withinRegion[['id_left','created_at','user_id','lon','lat','spatial_level', 'row_id', 'geometry','NUTS_ID', 'country_code']]
    withinRegion = withinRegion.rename(columns={'NUTS_ID':'post_region'})
    outside = gdf[~gdf['row_id'].isin(withinRegion['row_id'])]

    outside = outside[['id','row_id','lat','lon','geometry']]
    outside['post_region'] = outside.apply(lambda row: getClosestRegion(row), axis=1)
    print('Difference calculation done.')
    del gdf
    print('gdf deleted.')

    print('Region set.')
    withinRegion = withinRegion[['row_id','post_region']]
    outside = outside[['row_id','post_region']]
    withinRegion = pd.DataFrame(withinRegion)
    outside = pd.DataFrame(outside)
    result = pd.concat([withinRegion,outside])
    del withinRegion
    del outside
    print('Ready to upload')
    new_conn = mypool.connect()
    execute_values(new_conn, result, table)
    new_conn.close()
    endtime = datetime.now()
    print(f'Batch done! Batch took: {endtime-starttime}')
    return 'Finished'

a_pool = mp.Pool(8)
result = a_pool.map(postRegionAssign, range(0,last_tweet, batch_size))
a_pool.close()
a_pool.join()

final_time = datetime.now()- script_start
print(f"Script finished in :{final_time}")