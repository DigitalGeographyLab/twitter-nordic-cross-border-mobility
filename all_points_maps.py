import contextily as ctx
import csv
from datetime import datetime
import db_connection as db_con
import geojson
import geopandas as gpd
import folium
import io
from io import StringIO
import json
from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
from sqlalchemy import create_engine, func, distinct
import sys
import tempfile
starttime = datetime.now()

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
# Import credential and connection settings
db_name = db_con.db_name
db_username = db_con.db_username
db_host = db_con.db_host
db_port = db_con.db_port
db_pwd = db_con.db_pwd
engine_string = f"postgresql://{db_username}:{db_pwd}@{db_host}:{db_port}/{db_name}"
db_engine = create_engine(engine_string)


# SQL-query
query = 'SELECT id, lon,lat FROM pre_covid'
# Read data to dataframe
pre_points = db_con.read_sql_inmem_uncompressed(query, db_engine)

fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
pre_points_gdf = gpd.GeoDataFrame(pre_points, geometry=gpd.points_from_xy(pre_points['lon'], pre_points['lat']))
pre_points_gdf.crs = "EPSG:4326"
pre_points_gdf = pre_points_gdf.to_crs(epsg=3035)
pre_points_gdf.to_crs(epsg=3035).plot(ax=ax, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot

ax.axis("off")
ctx.add_basemap(ax,crs=pre_points_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

plt.tight_layout()
plt.savefig('imgs/nordic_all_pre_points.png', dpi=100)

del pre_points

query = 'SELECT id, lon,lat FROM post_covid'
# Read data to dataframe
post_points = db_con.read_sql_inmem_uncompressed(query, db_engine)

fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
post_points_gdf = gpd.GeoDataFrame(post_points, geometry=gpd.points_from_xy(post_points['lon'], post_points['lat']))
post_points_gdf.crs = "EPSG:4326"
post_points_gdf = post_points_gdf.to_crs(epsg=3035)
post_points_gdf.to_crs(epsg=3035).plot(ax=ax, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot

ctx.add_basemap(ax,crs=pre_points_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax.axis("off")


plt.tight_layout()
plt.savefig('imgs/nordic_all_post_points.png', dpi=100)

del post_points

fig, axes = plt.subplots(ncols = 2, figsize=(20,16))

ax11 = axes[0]
ax12 = axes[1]

pre_points_gdf.to_crs(epsg=3035).plot(ax=ax11, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot
post_points_gdf.to_crs(epsg=3035).plot(ax=ax12, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot

ctx.add_basemap(ax11,crs=pre_points_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax12,crs=pre_points_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax11.axis("off")
ax12.axis("off")
ax11.set_title('Pre-COVID-19',)
ax12.set_title('Post-COVID-19',)
plt.tight_layout()
plt.savefig('imgs/all_points.png', dpi=100)
print(f'Script took: {datetime.now()-starttime}')