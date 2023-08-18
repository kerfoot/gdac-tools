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
from gdt.erddap.osmc import OsmcClient
from gdt.plotting.calendars import plot_calendar
from pprint import pprint as pp

# Set up logger
log_state = 'INFO'
log_level = getattr(logging, log_state)
log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(format=log_format, level=log_level)

dataset_ids = ['sg677-20230530T0000']
img_path = '/Users/kerfoot/Downloads'
img_path = None
title = ''
plot_type = 'ymd'
clobber = True
response = 'csv'
pretty_print = 'psql'
include_summaries = False
debug = True
hours = 24
start_ts = None
end_ts = None
north = 90
south = -90
east = 180
west = -180
search_string = ''
variable_names = ''
glider = None
params = {}

client = GdacClient()

# Use the min and max times from allDatasets
start_time = client.erddap_datasets.mintime.min()
end_time = client.erddap_datasets.maxtime.max()

params = {}
if dataset_ids:
    logging.info('Searching for the following data sets:')
    for dataset_id in dataset_ids:
        logging.info('Dataset ID: {:}'.format(dataset_id))

else:

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
          'variableName': variable_names
          }

# Search the ERDDAP server
client.search_datasets(params=params, dataset_ids=dataset_ids)

datasets = client.datasets

if glider:
    logging.info('Filtering search results for gliders starting with {:}'.format(glider))
    datasets = datasets[datasets.glider.str.startswith(glider)]

logging.info('Found {:} data sets matching search criteria'.format(datasets.shape[0]))

osmc_client = OsmcClient()
