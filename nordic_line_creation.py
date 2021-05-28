from datetime import datetime
import db_connection as db_con
import fiona
from fiona.crs import from_epsg
import folium
from geoalchemy2 import Geometry
import geopandas as gpd
import io
from io import StringIO
from math import cos, sin, asin, sqrt, radians
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle
import psycopg2
import psycopg2.extras as extras
from pyproj import CRS
from shapely import wkt
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
import sqlalchemy.pool as pool
from sqlalchemy import create_engine, func, distinct
import sys

starttime = datetime.now()

def calc_distance(lat1, lon1, lat2, lon2):
    
    """
    Calculate the great circle distance between two points
    on Earth (specified in decimal degrees)
    """
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def getDistinctValues(column_name ,table_name, con):
    # Start cursor
    cur = con.cursor()
    # SQL string
    columnFind = f"SELECT {column_name} FROM {table_name} GROUP BY {column_name};"
    # Run SQL
    cur.execute(columnFind)
    # Get all results into a list
    values = [r[0] for r in cur.fetchall()]
    # Close cursor
    cur.close()
    return values

table = 'pre_covid'
users = getDistinctValues('user_id',table, db_con.psyco_con)

print(f'Number of users: {len(users)}')

all_lines = pd.DataFrame(columns=['geometry', 'user_id', 'orig_country', 'orig_time' , 'dest_time', 'duration', 'region_move', 'cb_move', 'distance_km'])

def getconn():
    c = psycopg2.connect(user=db_con.db_username, host=db_con.db_host, dbname=db_con.db_name, password=db_con.db_pwd)
    return c

mypool = pool.QueuePool(getconn, max_overflow=10, pool_size=30)


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


def shift_userLines(user):
    line_data = gpd.GeoDataFrame(columns=['geometry', 'user_id',  'dest_country', 'orig_time' , 'dest_time',  'duration','post_region','dest_region','region_move', 'cb_move', 'distance_km'], geometry='geometry')
    line_data = line_data.set_crs(epsg=4326)
    query1 = f'SELECT id,row_id, user_id, created_at, lat,lon, country_code, post_region FROM {table} WHERE user_id = {user} ORDER BY row_id'
    conn= mypool.connect()
    user1 = multi_read_sql_inmem_uncompressed(query1, conn)
    user1['created_at'] = user1['created_at'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S+00"))
    user1 = user1.rename(columns={'created_at': 'orig_time'})
    user1 = gpd.GeoDataFrame(user1, geometry=gpd.points_from_xy(user1.lon,user1.lat))
    user1 = user1.sort_values(by='orig_time')

    # create shifted columns
    user1['geo2'] = user1['geometry'].shift(-1)
    user1['dest_country'] = user1['country_code'].shift(-1)
    user1['dest_region'] = user1['post_region'].shift(-1)
    user1['dest_time'] = user1['orig_time'].shift(-1)
    user1['cb_move'] = user1.apply(lambda x: str(x['country_code'])+'-'+str(x['dest_country']) ,axis=1)
    user1['region_move'] = user1.apply(lambda x: str(x['post_region'])+'-'+str(x['dest_region']),axis=1)
    user1['duration'] = user1.apply(lambda x: (x['dest_time']-x['orig_time']).days,axis=1)
    
    # drop last row as last row has no value in geo2 due to shifting
    user1 = user1[:-1]
    if len(user1) <1:
        return line_data
    else:
        # create linestrings
        user1['line'] = user1.apply(lambda x: LineString([x['geometry'], x['geo2']]), axis=1)
        # replace point geometry with linestring
        user1['geometry'] = user1['line']
        # calculate distance
        user1['distance_km'] = user1['line'].apply(lambda line: calc_distance(line.xy[1][0], line.xy[0][0], line.xy[1][1], line.xy[0][1]))
        # clean up dataframe columns
        user1 = user1.drop(columns=['geo2','line'])
        # append individual values to line_data
        line_data = line_data.append(user1, ignore_index=True)

        line_data['orig_time'] = line_data['orig_time'].apply(lambda x: x.strftime("%Y-%m-%d-%H"))
        line_data['dest_time'] = line_data['dest_time'].apply(lambda x: x.strftime("%Y-%m-%d-%H"))
        
        return line_data



a_pool = mp.Pool(14)
result = a_pool.map(shift_userLines, users)
a_pool.close()
a_pool.join()

print('All user lines created.')

# Concat dataframes to one dataframe
all_lines = pd.concat(result, ignore_index=True)

print('All lines joined.')

def parallelize_dataframe(df, func, n_cores=2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def funcToRun(all_lines):
    all_lines['post_region'] = all_lines['post_region'].astype(str)
    all_lines['dest_region'] = all_lines['dest_region'].astype(str)
    all_lines['geometry'] = all_lines['geometry'].astype(str)
    return all_lines

all_lines = parallelize_dataframe(all_lines, funcToRun, 15)
all_lines = all_lines.dropna(subset=['geometry'])

conn = db_con.psyco_con

all_lines.to_sql('pre_covid_lines', con=db_con.db_engine, index=False, if_exists='replace')
print(f'Number of lines: {len(all_lines)}')

# Repeat with post_covid dataset

print(f'Script took: {datetime.now()-starttime}')
