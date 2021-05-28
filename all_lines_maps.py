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
query = 'SELECT id, row_id, geometry,cb_move FROM pre_covid_lines'
# Read data to dataframe
pre_lines = db_con.read_sql_inmem_uncompressed(query, db_engine)
# Apply wkt
pre_lines['geometry'] = pre_lines['geometry'].apply(wkt.loads)
# Convert to GeoDataFrame
pre_lines_gdf = gpd.GeoDataFrame(pre_lines, geometry='geometry')
# CRS
pre_lines_gdf.crs = "EPSG:4326"
pre_lines_gdf = pre_lines_gdf.to_crs(epsg=3035)
# Delete dataframe
del pre_lines
fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
pre_lines_gdf.to_crs(epsg=3035).plot(ax=ax, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot

ax.axis("off")
ctx.add_basemap(ax,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

plt.tight_layout()
plt.savefig('imgs/nordic_all_pre_lines.png', dpi=100)
#del pre_lines_gdf

# Pre Heatmaps
fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
pre_lines_gdf.to_crs(epsg=3035).plot(ax=ax, color='blue', edgecolor='black', linewidth=0.2, alpha=0.01) # 2 - Projected plot

ax.axis("off")
ctx.add_basemap(ax,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

plt.tight_layout()
plt.savefig('imgs/nordic_all_pre_lines_heat.png', dpi=100)

# COVID-19 data

# SQL-query
query = 'SELECT id, row_id, geometry, cb_move FROM post_covid_lines'
# Read data to dataframe
post_lines = db_con.read_sql_inmem_uncompressed(query, db_engine)
# Apply wkt
post_lines['geometry'] = post_lines['geometry'].apply(wkt.loads)
# Convert to GeoDataFrame
post_lines_gdf = gpd.GeoDataFrame(post_lines, geometry='geometry')
# CRS
post_lines_gdf.crs = "EPSG:4326"
post_lines_gdf = post_lines_gdf.to_crs(epsg=3035)
# Delete dataframe
del post_lines

# Plot all lines
fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
post_lines_gdf.to_crs(epsg=3035).plot(ax=ax, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot
ctx.add_basemap(ax,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax.axis("off")
plt.tight_layout()
plt.savefig('imgs/nordic_all_post_lines.png', dpi=100)


fig, axes = plt.subplots(ncols = 2, figsize=(20,16))

ax11 = axes[0]
ax12 = axes[1]

pre_lines_gdf.to_crs(epsg=3035).plot(ax=ax11, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot
post_lines_gdf.to_crs(epsg=3035).plot(ax=ax12, color='green', edgecolor='black', linewidth=0.3, alpha=0.5) # 2 - Projected plot

ctx.add_basemap(ax11,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax12,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax11.axis("off")
ax12.axis("off")
ax11.set_title('Pre-COVID-19',)
ax12.set_title('COVID-19',)
plt.tight_layout()
plt.savefig('imgs/all_lines.png', dpi=100)

# Plot All Travels Heatmap 
fig, ax = plt.subplots(ncols = 1, figsize=(20,16))

pre_lines_gdf.to_crs(epsg=3035).plot(ax=ax, color='blue', edgecolor='black', linewidth=0.2, alpha=0.01) # 2 - Projected plot
post_lines_gdf.to_crs(epsg=3035).plot(ax=ax, color='blue', edgecolor='black', linewidth=0.2, alpha=0.01) # 2 - Projected plot
xlim = ([pre_lines_gdf.total_bounds[0],  pre_lines_gdf.total_bounds[2]])

ylim = ([pre_lines_gdf.total_bounds[1],  5416499.996586122])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ctx.add_basemap(ax,crs=pre_lines_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax.axis("off")
ax.set_title('Heatmap All Travels Nordic Region',)
plt.tight_layout()
plt.savefig('imgs/all_lines_heat.png', dpi=100)

country_dict = {'FI':'Finland','DK':'Denmark','IS':'Iceland','NO':'Norway','SE':'Sweden'}
def country_heatmap(country):
    country_name = country_dict[country]
    all_pre_lines = pre_lines_gdf[pre_lines_gdf['cb_move'].str.contains(country)]
    all_post_lines = post_lines_gdf[post_lines_gdf['cb_move'].str.contains(country)]
    all_lines = all_pre_lines.append(all_post_lines)
    fig, ax = plt.subplots(ncols = 1, figsize=(20,16))
    all_lines.to_crs(epsg=3035).plot(ax=ax, color='blue', edgecolor='black', linewidth=0.2, alpha=0.01) # 2 - Projected plot
    xlim = ([all_lines.total_bounds[0],  all_lines.total_bounds[2]])
    #ylim = ([all_lines.total_bounds[1],  all_lines.total_bounds[3]])
    ylim = ([all_lines.total_bounds[1],  5416499.996586122])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ctx.add_basemap(ax,crs=all_lines.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    ax.axis("off")
    title_string = f"Heatmap All Travels To and From {country_name}"
    ax.set_title(title_string,)
    plt.tight_layout()
    file_string = f"{country}_heatmap.png"
    plt.savefig(f'imgs/heatmaps/{file_string}', transparent=True, dpi=100)
    print(f'{country_name} heatmap created')

country_heatmap_list = ['FI','DK','IS','NO','SE']

def heatmap_multi(setting):
    country_heatmap(setting)

a_pool = mp.Pool(15)
result = a_pool.map(country_heatmap, country_heatmap_list)
a_pool.close()
a_pool.join()

print(f'Script took: {datetime.now()-starttime}')