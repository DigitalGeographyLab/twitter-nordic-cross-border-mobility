from adjustText import adjust_text
import contextily as ctx
import csv
from datetime import datetime
import db_connection as db_con
import descartes
from descartes import PolygonPatch
import folium
from functools import partial
import geojson
import geopandas as gpd
from geopandas.tools import geocode
from geopy.geocoders import Nominatim
import io
from io import StringIO
import json
from matplotlib import cm
from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import matplotlib.patches as mpatches
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import operator
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import psycopg2
import psycopg2.extras as extras
import seaborn as sns
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
from sqlalchemy import create_engine, func, distinct
import sys
import tempfile

starttime = datetime.now()

cmap = mpl.cm.get_cmap('Blues')
border_color = cmap(0.75,0.7)

cmap = mpl.cm.get_cmap('Reds')
color_map = [cmap(0.05,1),cmap(.25,1),cmap(0.5,1),cmap(0.75,1),cmap(1,1)]
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

## Pre_lines
# SQL-query
query = 'SELECT geometry,user_id,dest_country,orig_time,dest_time,duration,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM pre_covid_lines'
# Read data to dataframe
pre_lines = db_con.read_sql_inmem_uncompressed(query, db_engine)
# Apply wkt
pre_lines['geometry'] = pre_lines['geometry'].apply(wkt.loads)
# Convert to GeoDataFrame
pre_lines_gdf = gpd.GeoDataFrame(pre_lines, geometry='geometry')
# CRS
pre_lines_gdf.crs = "EPSG:4326"
# Delete dataframe
del pre_lines
# Convert timestamps
pre_lines_gdf['orig_time'] = pre_lines_gdf['orig_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
pre_lines_gdf['dest_time'] = pre_lines_gdf['dest_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
# Add month
pre_lines_gdf['month'] = pre_lines_gdf['dest_time'].apply(lambda x: x.month)
pre_lines_gdf['yearmonth'] = pre_lines_gdf['dest_time'].apply(lambda x: int(str(x.year)+str(x.month).zfill(2)))


## Post_lines
# SQL-query
query = 'SELECT geometry,user_id,dest_country,orig_time,dest_time,duration,post_region,dest_region,region_move,cb_move,distance_km,row_id,country_code FROM post_covid_lines'
# Read data to dataframe
post_lines = db_con.read_sql_inmem_uncompressed(query, db_engine)
# Apply wkt
post_lines['geometry'] = post_lines['geometry'].apply(wkt.loads)
# Convert to GeoDataFrame
post_lines_gdf = gpd.GeoDataFrame(post_lines, geometry='geometry')
# CRS
post_lines_gdf.crs = "EPSG:4326"
# Delete dataframe
del post_lines
post_lines_gdf['orig_time'] = post_lines_gdf['orig_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
post_lines_gdf['dest_time'] = post_lines_gdf['dest_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d-%H"))
# Add month
post_lines_gdf['month'] = post_lines_gdf['dest_time'].apply(lambda x: x.month)
post_lines_gdf['yearmonth'] = post_lines_gdf['dest_time'].apply(lambda x: int(str(x.year)+str(x.month).zfill(2)))

# Cities to plot
cities_list = ['Copenhagen, Denmark','Malmö, Sweden','Oslo, Norway','Helsinki, Finland','Stockholm, Sweden','Tornio, Finland','Haparanda, Sweden','Göteborg, Sweden', 'Tromsø, Norway','Bergen, Norway', 'Århus, Denmark','Stavanger,Norway','Reykjavik, Iceland', 'Trondheim, Norway', ' Vasa, Bangatan, Finland','Umeå, Sweden','Oulu, Finland','Sundsvall, Sweden','Turku, Finland', 'Kiruna, Sweden','Ålborg, Denmark','Luleå, Sweden', 'Rovaniemi, Finland', 'Strömstad, Sweden', 'Halden, Norway', 'Alta, Norway']
df = pd.DataFrame(cities_list)
geolocator = Nominatim(user_agent="haavard")
geocode2 = partial(geolocator.geocode, language="en")
cities_df = geocode(df[0], provider='nominatim', user_agent='haavard', timeout=4)
cities_df = cities_df.to_crs(epsg=3035)
cities_df['x'] = cities_df['geometry'].apply(lambda point: point.x)
cities_df['y'] = cities_df['geometry'].apply(lambda point: point.y)
cities_df['label'] = cities_df['address'].apply(lambda x: geocode2(x))
cities_df['label'] = cities_df['label'].apply(lambda x: list(str(x).split(","))[0])
cities_df['label'] = cities_df['label'].apply(lambda x: list(str(x).split(" "))[0])
del df

## Nordic Countries
nordic_and_baltics_geopackage_fp = 'NordicBalticsIceland.gpkg'
nordics_and_balt = gpd.read_file(nordic_and_baltics_geopackage_fp)
nordics = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match('DK|FI|NO|SE|IS')]
regions_fp = 'NUTS2.gpkg'
regions = gpd.read_file(regions_fp)
regions = regions.to_crs(epsg=3035)

def functional_area_countries(country_one, country_two, pre_or_post):
	countries_string = f"{country_one}-{country_two}|{country_two}-{country_one}"
	time_list = ['pre','post']
	if pre_or_post == 'pre':
		all_lines = pre_lines_gdf[pre_lines_gdf['cb_move'].str.match(countries_string)]
		title_string = f"Pre-COVID-19 {country_one}-{country_two} Functional Area"
	else:
		all_lines = post_lines_gdf[post_lines_gdf['cb_move'].str.match(countries_string)]
		title_string = f"COVID-19 {country_one}-{country_two} Functional Area"
	all_lines['start_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][0], axis=1)
	all_lines['end_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][1], axis=1)
	# country one
	one_starts = all_lines[all_lines['country_code'] == country_one]
	one_starts = one_starts[['dest_country','country_code', 'start_point','row_id']]
	one_starts['geometry'] = one_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	one_ends = all_lines[all_lines['dest_country'] == country_one]
	one_ends = one_ends[['dest_country','country_code', 'end_point','row_id']]
	one_ends['geometry'] = one_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	one = one_starts.append(one_ends)
	del one_starts
	del one_ends
	one = one.set_geometry('geometry')
	one.crs = "EPSG:4326"
	one = one.to_crs(epsg=3035)
	# country two
	two_starts = all_lines[all_lines['country_code'] == country_two]
	two_starts = two_starts[['dest_country','country_code', 'start_point','row_id']]
	two_starts['geometry'] = two_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	two_ends = all_lines[all_lines['dest_country'] == country_two]
	two_ends = two_ends[['dest_country','country_code', 'end_point','row_id']]
	two_ends['geometry'] = two_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	two = two_starts.append(two_ends)
	del two_starts
	del two_ends
	two = two.set_geometry('geometry')
	two.crs = "EPSG:4326"
	two = two.to_crs(epsg=3035)

	# set up total bounds
	country_one_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(country_one)]
	country_two_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(country_two)]
	total_bounds_gdf = country_one_clipping.append(country_two_clipping)
	# plot
	cmap = mpl.cm.get_cmap('Reds')
	color_map = [cmap(0.05,0.8),cmap(.25,0.8),cmap(0.5,0.8),cmap(0.75,0.8),cmap(1,0.8)]

	f, ax = plt.subplots(ncols=1, figsize=(20, 16))

	on = sns.kdeplot(x=one['geometry'].x,y= one['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	on.collections[0].set_facecolor(color_map[0])
	on.collections[1].set_facecolor(color_map[1])
	on.collections[2].set_facecolor(color_map[2])
	on.collections[3].set_facecolor(color_map[3])

	p = PolygonPatch(country_one_clipping['geometry'].iloc[0],transform=ax.transData)
	num_coll = len(ax.collections)
	for col in ax.collections[:num_coll]:
		col.set_clip_path(p)

	tw = sns.kdeplot(x=two['geometry'].x,y= two['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	tw.collections[num_coll+0].set_facecolor(color_map[0])
	tw.collections[num_coll+1].set_facecolor(color_map[1])
	tw.collections[num_coll+2].set_facecolor(color_map[2])
	tw.collections[num_coll+3].set_facecolor(color_map[3])
	q = PolygonPatch(country_two_clipping['geometry'].iloc[0],transform=ax.transData)

	for col in ax.collections[num_coll:]:
		col.set_clip_path(q)

	cities = gpd.sjoin(cities_df,total_bounds_gdf, how='inner', op='within')
	cit = cities.plot(ax=ax,color='black',markersize=10, zorder=6)
	texts = [cit.text(cities['x'].iloc[i], cities['y'].iloc[i], cities['label'].iloc[i], ha='center', va='center',fontsize=18,zorder=7) for i in range(len(cities))]
	adjust_text(texts)
	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	ax.set_axis_off()

	xlim = ([total_bounds_gdf.total_bounds[0],  total_bounds_gdf.total_bounds[2]])
	ylim = ([total_bounds_gdf.total_bounds[1],  total_bounds_gdf.total_bounds[3]])

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.tight_layout()
	patch_one = mpatches.Patch(color=color_map[0], label='95%')
	patch_two = mpatches.Patch(color=color_map[1], label='75%')
	patch_three = mpatches.Patch(color=color_map[2], label='50%')
	patch_four = mpatches.Patch(color=color_map[3], label='25%')
	plt.legend(handles=[patch_one,patch_two,patch_three,patch_four],title="Share of Travels")
	ax.set_title(title_string,)
	file_string = f"{pre_or_post}_{country_one}_{country_two}_functional_area.png"
	plt.savefig(f'imgs/functional_area/{file_string}',dpi=300)
	#plt.show()
	print(f'{countries_string} {pre_or_post} map created')
	return file_string


list_of_settings = [('DK','SE','pre'),('DK','SE','post'),('DK','NO','pre'),('DK','NO','post'),('FI','SE','pre'),('FI','SE','post'),('FI','NO','pre'),('FI','NO','post'),('NO','SE','pre'),('NO','SE','post'),('NO','IS','pre'),('NO','IS','post'),('DK','IS','pre'),('DK','IS','post'),('DK','FI','pre'),('DK','FI','post')]
def func(threeple):
	functional_area_countries(threeple[0],threeple[1],threeple[2])
a_pool = mp.Pool(15)
result = a_pool.map(func, list_of_settings)
a_pool.close()
a_pool.join()

def functional_area_regions(region_one, region_two, pre_or_post):
	regions_string = f"{region_one}-{region_two}|{region_two}-{region_one}"
	time_list = ['pre','post']
	if pre_or_post == 'pre':
		all_lines = pre_lines_gdf[pre_lines_gdf['region_move'].str.match(regions_string)]
		title_string = f"Pre-COVID-19 {region_one}-{region_two} Functional Area"
	else:
		all_lines = post_lines_gdf[post_lines_gdf['region_move'].str.match(regions_string)]
		title_string = f"COVID-19 {region_one}-{region_two} Functional Area"
	all_lines['start_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][0], axis=1)
	all_lines['end_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][1], axis=1)
	# country one
	one_starts = all_lines[all_lines['post_region'] == region_one]
	one_starts = one_starts[['dest_region','post_region', 'start_point','row_id']]
	one_starts['geometry'] = one_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	one_ends = all_lines[all_lines['dest_region'] == region_one]
	one_ends = one_ends[['dest_region','post_region', 'end_point','row_id']]
	one_ends['geometry'] = one_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	one = one_starts.append(one_ends)
	del one_starts
	del one_ends
	one = one.set_geometry('geometry')
	one.crs = "EPSG:4326"
	one = one.to_crs(epsg=3035)
	# country two
	two_starts = all_lines[all_lines['post_region'] == region_two]
	two_starts = two_starts[['dest_region','post_region', 'start_point','row_id']]
	two_starts['geometry'] = two_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	two_ends = all_lines[all_lines['dest_region'] == region_two]
	two_ends = two_ends[['dest_region','post_region', 'end_point','row_id']]
	two_ends['geometry'] = two_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	two = two_starts.append(two_ends)
	del two_starts
	del two_ends
	two = two.set_geometry('geometry')
	two.crs = "EPSG:4326"
	two = two.to_crs(epsg=3035)

	# set up total bounds
	region_one_country = region_one[:2]
	region_two_country = region_two[:2]
	country_one_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(region_one_country)]
	country_two_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(region_two_country)]
	region_one_clipping = regions[regions['NUTS_ID'].str.match(region_one)]
	region_two_clipping = regions[regions['NUTS_ID'].str.match(region_two)]
	total_bounds_gdf = region_one_clipping.append(region_two_clipping)

	# plot
	f, ax = plt.subplots(ncols=1, figsize=(20, 16))
	total_bounds_gdf.plot(ax=ax, color=None)
	on = sns.kdeplot(x=one['geometry'].x,y= one['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	on.collections[0].set_facecolor(color_map[0])
	on.collections[1].set_facecolor(color_map[1])
	on.collections[2].set_facecolor(color_map[2])
	on.collections[3].set_facecolor(color_map[3])
	on.collections[4].set_facecolor(color_map[3])
	a = PolygonPatch(country_one_clipping['geometry'].iloc[0],transform=ax.transData)
	p = PolygonPatch(region_one_clipping['geometry'].iloc[0],transform=ax.transData)
	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	num_coll = len(ax.collections)
	for col in ax.collections[:num_coll]:
		col.set_clip_path(p)
	
	tw = sns.kdeplot(x=two['geometry'].x,y= two['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	tw.collections[num_coll+0].set_facecolor(color_map[0])
	tw.collections[num_coll+1].set_facecolor(color_map[1])
	tw.collections[num_coll+2].set_facecolor(color_map[2])
	tw.collections[num_coll+3].set_facecolor(color_map[3])
	b = PolygonPatch(country_two_clipping['geometry'].iloc[0],transform=ax.transData)
	q = PolygonPatch(region_two_clipping['geometry'].iloc[0],transform=ax.transData)
	for col in ax.collections[num_coll:]:
		col.set_clip_path(q)

	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	ax.set_axis_off()

	xlim = ([total_bounds_gdf.total_bounds[0],  total_bounds_gdf.total_bounds[2]])
	ylim = ([total_bounds_gdf.total_bounds[1],  total_bounds_gdf.total_bounds[3]])
	
	
	r = region_one_clipping.plot(ax=ax,facecolor="none", edgecolor=border_color, zorder=5, transform=ax.transData)
	s = region_two_clipping.plot(ax=ax,facecolor="none", edgecolor=border_color, zorder=5, transform=ax.transData)
	cities = gpd.sjoin(cities_df,total_bounds_gdf, how='inner', op='within')
	cities.plot(ax=ax,color='black',markersize=10, zorder=6)
	cit = cities.plot(ax=ax,color='black',markersize=10, zorder=6)
	texts = [cit.text(cities['x'].iloc[i], cities['y'].iloc[i], cities['label'].iloc[i], ha='center', va='center',fontsize=18,zorder=7) for i in range(len(cities))]
	adjust_text(texts)
	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.tight_layout()
	patch_one = mpatches.Patch(color=color_map[0], label='95%')
	patch_two = mpatches.Patch(color=color_map[1], label='75%')
	patch_three = mpatches.Patch(color=color_map[2], label='50%')
	patch_four = mpatches.Patch(color=color_map[3], label='25%')
	plt.legend(handles=[patch_one,patch_two,patch_three,patch_four],title="Share of Travels")
	ax.set_title(title_string,)
	file_string = f"{pre_or_post}_{region_one}_{region_two}_functional_area.png"
	plt.savefig(f'imgs/functional_area/regional/{file_string}',dpi=300)
	print(f'{regions_string} {pre_or_post} map created')
	return file_string


region_settings = [('NO07','FI1D','pre'),('NO07','FI1D','post'),('DK01','SE22','pre'),('DK01','SE22','post'),('SE33','FI1D','pre'),('SE33','FI1D','post'), ('NO08','SE23','pre'),('NO08','SE23','post')]
def region_multi_func(setting):
	functional_area_regions(setting[0],setting[1],setting[2])
a_pool = mp.Pool(15)
result = a_pool.map(region_multi_func, region_settings)
a_pool.close()
a_pool.join()



def functional_area_multi_regions(regions_country_one, regions_country_two, pre_or_post):
	country_one = regions_country_one[0][:2]
	country_two = regions_country_two[0][:2]
	region_pairs = []
	for region in regions_country_one:
		for reg in range(len(regions_country_two)):
			appendstring = f"{region}-{regions_country_two[reg]}"
			reverse_append = f"{regions_country_two[reg]}-{region}"
			region_pairs.append(appendstring)
			region_pairs.append(reverse_append)
	
	regions_string = '|'.join(region_pairs)

	time_list = ['pre','post']
	if pre_or_post == 'pre':
		all_lines = pre_lines_gdf[pre_lines_gdf['region_move'].str.match(regions_string)]
		title_string = f"Pre-COVID-19 {country_one}-{country_two} Multi Regional Functional Area"
	else:
		all_lines = post_lines_gdf[post_lines_gdf['region_move'].str.match(regions_string)]
		title_string = f"COVID-19 {country_one}-{country_two} Multi Regional Functional Area"
	
	all_lines['start_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][0], axis=1)
	all_lines['end_point'] = all_lines.apply(lambda x: [y for y in x['geometry'].coords][1], axis=1)
	
	# country one
	one_starts = all_lines[all_lines['country_code'] == country_one]
	one_starts = one_starts[['dest_country','country_code', 'start_point','row_id']]
	one_starts['geometry'] = one_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	one_ends = all_lines[all_lines['dest_country'] == country_one]
	one_ends = one_ends[['dest_country','country_code', 'end_point','row_id']]
	one_ends['geometry'] = one_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	one = one_starts.append(one_ends)
	del one_starts
	del one_ends
	one = one.set_geometry('geometry')
	one.crs = "EPSG:4326"
	one = one.to_crs(epsg=3035)
	# country two
	two_starts = all_lines[all_lines['country_code'] == country_two]
	two_starts = two_starts[['dest_country','country_code', 'start_point','row_id']]
	two_starts['geometry'] = two_starts['start_point'].apply(lambda x: Point(x[0],x[1]))
	two_ends = all_lines[all_lines['dest_country'] == country_two]
	two_ends = two_ends[['dest_country','country_code', 'end_point','row_id']]
	two_ends['geometry'] = two_ends['end_point'].apply(lambda x: Point(x[0],x[1]))
	two = two_starts.append(two_ends)
	del two_starts
	del two_ends
	two = two.set_geometry('geometry')
	two.crs = "EPSG:4326"
	two = two.to_crs(epsg=3035)

	# set up total bounds
	country_one_regions_string = '|'.join(regions_country_one)
	country_one_regions = regions[regions['NUTS_ID'].str.match(country_one_regions_string)]
	country_one_regions = country_one_regions.dissolve(by='CNTR_ID')
	country_two_regions_string = '|'.join(regions_country_two)
	country_two_regions = regions[regions['NUTS_ID'].str.match(country_two_regions_string)]
	country_two_regions = country_two_regions.dissolve(by='CNTR_ID')

	country_one_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(country_one)]
	country_two_clipping = nordics_and_balt[nordics_and_balt['CNTR_ID'].str.match(country_two)]
	
	total_bounds_gdf = country_one_regions.append(country_two_regions)


	# plot
	f, ax = plt.subplots(ncols=1, figsize=(20, 16))
	total_bounds_gdf.plot(ax=ax, color=None)
	on = sns.kdeplot(x=one['geometry'].x,y= one['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	on.collections[0].set_facecolor(color_map[0])
	on.collections[1].set_facecolor(color_map[1])
	on.collections[2].set_facecolor(color_map[2])
	on.collections[3].set_facecolor(color_map[3])
	on.collections[4].set_facecolor(color_map[3])
	a = PolygonPatch(country_one_clipping['geometry'].iloc[0],transform=ax.transData)
	p = PolygonPatch(country_one_regions['geometry'].iloc[0],transform=ax.transData)
	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	num_coll = len(ax.collections)
	for col in ax.collections[:num_coll]:
		col.set_clip_path(p)
	
	tw = sns.kdeplot(x=two['geometry'].x,y= two['geometry'].y, shade=True, cmap='Reds',alpha=0.9, ax=ax, levels=[0.05,0.25,0.50,0.75,1])
	tw.collections[num_coll+0].set_facecolor(color_map[0])
	tw.collections[num_coll+1].set_facecolor(color_map[1])
	tw.collections[num_coll+2].set_facecolor(color_map[2])
	tw.collections[num_coll+3].set_facecolor(color_map[3])
	b = PolygonPatch(country_two_clipping['geometry'].iloc[0],transform=ax.transData)
	q = PolygonPatch(country_two_regions['geometry'].iloc[0],transform=ax.transData)
	for col in ax.collections[num_coll:]:
		col.set_clip_path(q)

	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	ax.set_axis_off()

	xlim = ([total_bounds_gdf.total_bounds[0],  total_bounds_gdf.total_bounds[2]])
	ylim = ([total_bounds_gdf.total_bounds[1],  total_bounds_gdf.total_bounds[3]])
	
	
	r = country_one_regions.plot(ax=ax,facecolor="none", edgecolor=border_color, zorder=5, transform=ax.transData)
	s = country_two_regions.plot(ax=ax,facecolor="none", edgecolor=border_color, zorder=5, transform=ax.transData)
	cities = gpd.sjoin(cities_df,total_bounds_gdf, how='inner', op='within')
	cit = cities.plot(ax=ax,color='black',markersize=10, zorder=6)
	texts = [cit.text(cities['x'].iloc[i], cities['y'].iloc[i], cities['label'].iloc[i], ha='center', va='center',fontsize=18,zorder=7) for i in range(len(cities))]
	adjust_text(texts)
	ctx.add_basemap(ax=ax,crs = total_bounds_gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels) # 3
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.tight_layout()
	patch_one = mpatches.Patch(color=color_map[0], label='95%')
	patch_two = mpatches.Patch(color=color_map[1], label='75%')
	patch_three = mpatches.Patch(color=color_map[2], label='50%')
	patch_four = mpatches.Patch(color=color_map[3], label='25%')
	plt.legend(handles=[patch_one,patch_two,patch_three,patch_four],title="Share of Travels")
	ax.set_title(title_string,)
	file_string = f"{pre_or_post}_{country_one}_{regions_country_one[0]}_{country_two}_{regions_country_two[0]}_functional_area.png"
	plt.savefig(f'imgs/functional_area/multi_regional/{file_string}',dpi=300)
	#plt.show()
	print(f'{regions_string} {pre_or_post} map created')
	return file_string


multi_region_settings = [(['DK01','DK02','DK03','DK04','DK05'],['SE22','SE23','SE21'],'pre'),(['DK01','DK02','DK03','DK04','DK05'],['SE22','SE23','SE21'],'post'),(['DK01','DK02','DK03','DK04','DK05'],['NO02','NO08','NO09','NO0A'],'pre'),(['DK01','DK02','DK03','DK04','DK05'],['NO02','NO08','NO09','NO0A'],'post'),(['NO02','NO08','NO09','NO0A'],['SE23','SE11','SE12','SE22','SE21','SE31'],'pre'),(['NO02','NO08','NO09','NO0A'],['SE23','SE11','SE12','SE22','SE21','SE31'],'post'),(['NO06','NO07'],['SE32','SE33'],'pre'),(['NO06','NO07'],['SE32','SE33'],'post'),(['SE33','SE32'],['FI1D','FI19'],'pre'),(['SE33','SE32'],['FI1D','FI19'],'post'),(['SE12','SE11','SE23','SE22','SE31','SE21'],['FI1B','FI1C'],'pre'),(['SE12','SE11','SE23','SE22','SE31','SE21'],['FI1B','FI1C'],'post'),(['SE12','SE11','SE23','SE22','SE31','SE21'],['DK01','DK02','DK03','DK04','DK05'],'pre'),(['SE12','SE11','SE23','SE22','SE31','SE21'],['DK01','DK02','DK03','DK04','DK05'],'post')]

def multi_region_multi_func(setting):
	functional_area_multi_regions(setting[0],setting[1],setting[2])

a_pool = mp.Pool(15)
result = a_pool.map(multi_region_multi_func, multi_region_settings)
a_pool.close()
a_pool.join()




print(f'Script took: {datetime.now()-starttime}')