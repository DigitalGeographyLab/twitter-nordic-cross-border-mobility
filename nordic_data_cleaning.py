import csv
from datetime import datetime
import db_connection as db_con
from fileinput import FileInput
import geojson
import geopandas as gpd
from glob import glob
import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import operator
import os
from os.path import isdir, isfile, join
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from shapely.geometry import Point, LineString, Polygon
from sqlalchemy import create_engine
import sqlalchemy.pool as pool

start_time = datetime.now()

def getconn():
    c = psycopg2.connect(user=db_con.db_username, host=db_con.db_host, dbname=db_con.db_name, password=db_con.db_pwd)
    return c
mypool = pool.QueuePool(getconn, max_overflow=10, pool_size=25)

alchemyEngine = create_engine(f'postgresql://{db_con.db_username}:{db_con.db_pwd}@localhost/{db_con.db_name}', echo = False)

def setup_connection():
    passwd = db_con.db_pwd
    db_name = db_con.db_name
    db_user = db_con.db_username
    alchemyEngine = create_engine(f'postgresql://{db_user}:{passwd}@localhost/{db_name}', pool_size=20, echo = False)
    connection = alchemyEngine.connect()
    return(connection)


path = 'premiumsearcher/data/pre_covid'
f_list = glob(join(path,'*.json'))

def addJSON(f_name):
    connection = setup_connection()
    try:
        print(f'Starting with file {f_name}')
        #make a pandas df out of the file contents:
        tweets_df = pd.read_json(f_name, lines = True)
        print('DataFrame created')
        # Select columns for subsetting
        columns = ['id','id_str','created_at','user.id','geo.coordinates.coordinates','geo.bbox','geo.place_type','geo.country_code', 'geo.centroid.x', 'geo.centroid.y', 'geo.coordinates.x','geo.coordinates.y']
        # Subset data
        subset = tweets_df.filter(items=columns).rename(columns={'user.id':'user_id','geo.bbox':'bounding_box','geo.place_type': 'place_type','geo.country_code':'country_code', 'geo.coordinates.coordinates': 'geo_coordinates'})
        subset['id_str'] = subset.apply(lambda row: str(row['id']), axis=1)
        # Add latitude
        subset['lat'] = subset.apply(lambda row: row['geo.coordinates.y'] if type(row['geo_coordinates']) == list else row['geo.centroid.y'], axis=1)
        # Add longitude
        subset['lon'] = subset.apply(lambda row: row['geo.coordinates.x'] if type(row['geo_coordinates']) == list else row['geo.centroid.x'], axis=1)
        # Set spatial level
        subset['spatial_level'] = subset.apply(lambda row: row['place_type'] if type(row['geo_coordinates']) == list else 'centroid' , axis=1)
        # Add user_lat
        subset['user_lat'] = subset.apply(lambda row: 'None'  ,axis=1)
        # Add user_lon
        subset['user_lon'] = subset.apply(lambda row: 'None'  ,axis=1)
        # Add user country
        subset['user_country'] = subset.apply(lambda row: 'None'  ,axis=1)
        # Reformat and normalize date
        subset['created_at'] = subset['created_at'].apply(lambda x: pd.to_datetime(datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S%z"), utc=True))
        # Remove unneeded columns
        subset = subset.drop(['geo_coordinates','geo.coordinates.x','geo.coordinates.y','geo.centroid.x','geo.centroid.y'], axis=1)
        print("Changes made")
        subset.to_sql('pre_covid', connection, index=False, if_exists='append')
    except ValueError as vx:
        print(f'Trouble with File {f_name}:')
        print(vx)
    except Exception as ex:
        print(f'Trouble with File {f_name}:')
        print(ex)
    else:
        print(f'File {f_name} processed successfully')
    connection.close()
    return(f_name)


a_pool = mp.Pool(15)
result = a_pool.map(addJSON, f_list)
a_pool.close()
a_pool.join()
print(result)


print(f'Time used: {datetime.now()-start_time}')

#11:08:44.549273