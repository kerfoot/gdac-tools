"""
Load the gdt.erddap.GdacClient and search the glider DAC ERDDAP server
"""

import logging
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser
from gdt.erddap import GdacClient
from gdt.apis.dac import fetch_dac_catalog_dataframe
from gdt.urls import end_points
from gdt.plotting.calendars import plot_calendar
from pprint import pprint as pp

# Set up logger
log_state = 'INFO'
log_level = getattr(logging, log_state)
log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(format=log_format, level=log_level)

dataset_ids = []
clobber = True
response = 'csv'
pretty_print = 'psql'
include_summaries = False
debug = True
hours = 0
start_ts = None
end_ts = None
north = 90
south = -90
east = 180
west = -180
search_string = ''
variable_name = ''
glider = None
params = {}

client = GdacClient()

start_time = client.erddap_datasets.mintime.min()
end_time = client.erddap_datasets.maxtime.max()

if not dataset_ids:

    if start_ts:
        hours = 0
        try:
            start_time = parser.parse(start_ts)
            logging.info('User specified start_time: {:}'.format(start_time.strftime('%Y-%m-%dT%H:%M:%S')))
        except Exception as e:
            logging.error('User specified start_time: {:} ({:})'.format(start_ts, e))

    if end_ts:
        hours = 0
        try:
            end_time = parser.parse(end_ts)
            logging.info('User specified end_time: {:}'.format(end_time.strftime('%Y-%m-%dT%H:%M:%S')))
        except Exception as e:
            logging.error('User specified end_time: {:} ({:})'.format(end_ts, e))

    if hours > 0:
        logging.info('Searching for data sets updated within the last {:} hours'.format(hours))
        start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)

start_time = start_time.strftime('%Y-%m-%dT%H:%M')
end_time = end_time.strftime('%Y-%m-%dT%H:%M')
logging.info('Search start time: {:}'.format(start_time))
logging.info('Search end time  : {:}'.format(end_time))

params = {'min_time': start_time,
          'max_time': end_time,
          'min_lat': south,
          'max_lat': north,
          'min_lon': west,
          'max_lon': east,
          'search_for': search_string,
          'variableName': variable_name
          }

client.search_datasets(params=params, dataset_ids=dataset_ids)

datasets = client.datasets

if glider:
    logging.info('Filtering search results for gliders starting with {:}'.format(glider))
    datasets = datasets[datasets.glider.str.startswith(glider)]

logging.info('Found {:} data sets matching search criteria'.format(datasets.shape[0]))

# Get the data set registration API endpoint

api_df = fetch_dac_catalog_dataframe(end_points['NGDAC_API_DEPLOYMENTS_URL'].url)
drop_cols = ['wmo_id',
             'thredds',
             'sos',
             'iso',
             'erddap',
             'dap']
api_df = api_df.drop(columns=drop_cols)
# Extract the glider name from the index (dataset_id) and replace the glider_name
gliders = api_df.index.str.extract(r'^(.*)-\d{8}T\d{4,}')
api_df['glider_name'] = gliders[0].to_list()
# Rename the glider_name column to glider
api_df.rename(columns={'glider_name': 'glider'})

# Merge existing ERDDAP data sets with registered data set registration information
merged = client.datasets.drop(columns=['glider']).merge(api_df,
                                                        how='left',
                                                        right_index=True,
                                                        left_index=True)

# Create the orphaned column and set to True where there is not tabledap url
merged['orphaned'] = True
merged['orphaned'].where(merged.tabledap.isnull(), False, inplace=True)

# Merge API data sets with ERDDAP data sets
merged = client.datasets.drop(columns=['glider']).merge(api_df,
                                                        how='right',
                                                        right_index=True,
                                                        left_index=True)

# Create the orphaned column and set to True where there is not tabledap url
merged['orphaned'] = True
merged['orphaned'].where(merged.tabledap.isnull(), False, inplace=True)

# Remove delayed_mode data sets since we did not include those in the ERDDAP search
merged = merged[~merged.delayed_mode]

# Orphaned real-time data sets
orphaned = merged[merged.orphaned]
