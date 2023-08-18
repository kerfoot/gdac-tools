#!/usr/bin/env python

import sys
import argparse
import logging
import datetime
import pytz
import tabulate
import pandas as pd
import tabulate
from dateutil import parser
from gdt.erddap import GdacClient
from gdt.erddap.osmc import OsmcClient


def main(args):
    """Query the NOAA Observing System Monitoring Center (OSMC) ERDDAP server for available GTS glider observations"""
    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    dataset_ids = args.dataset_ids
    hours = args.hours
    start_ts = args.start_ts
    end_ts = args.end_ts
    clip_times = args.clip
    response = args.format
    debug = args.debug
    north = args.north
    south = args.south
    east = args.east
    west = args.west
    include_gps = args.gps
    search_string = args.search_string or ''
    glider = args.glider

    client = GdacClient()
    if client.erddap_datasets.empty:
        logging.info('ERDDAP server contains no data sets: {:}'.format(client.server))
        return 1

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
              'search_for': search_string
              }

    # Search
    client.search_datasets(params=params, dataset_ids=dataset_ids)

    if client.datasets.empty:
        logging.warning('No datasets found matching the search criteria')
        return 1

    datasets = client.datasets

    if glider:
        logging.info('Filtering search results for gliders starting with {:}'.format(glider))
        datasets = datasets[datasets.glider.str.startswith(glider)]

    logging.info('Found {:} DAC datasets matching search criteria'.format(datasets.shape[0]))

    osmc_client = OsmcClient()

    obs_counts = []
    logging.info('Checking OSMC for corresponding GTS observations')
    if include_gps:
        logging.info('Including GPS in GTS observation searches')
    else:
        logging.info('Omitting GPS in GTS observation searches')

    for dataset_id, row in datasets.iterrows():
        dataset_obs = {'dataset_id': dataset_id,
                       'wmo_id': row.wmo_id,
                       'start_date': row.start_date,
                       'end_date': row.end_date,
                       'dac_profiles_count': 0,
                       'osmc_gts_obs_count': 0}

        logging.info('Fetching GTS obs for {:}'.format(dataset_id))

        # Count the number of DAC profiles
        if dataset_ids or not clip_times:
            dataset_obs['dac_profiles_count'] = row.num_profiles
        else:
            logging.info('Fetching DAC profile count for {:}'.format(dataset_id))
            dac_profiles = client.get_dataset_profiles(dataset_id)
            dataset_obs['dac_profiles_count'] = dac_profiles.loc[start_time:end_time].shape[0]

        if row['wmo_id'] is None:
            logging.warning('Skipping GTS fetch for {:} (No WMO id)'.format(dataset_id))
            obs_counts.append(dataset_obs)
            continue

        # Get the available GTS obs
        obs = osmc_client.get_profiles_by_wmo_id(row['wmo_id'], row.start_date, row.end_date, gps=include_gps)

        if obs.empty:
            logging.warning(
                'No GTS observations found for dataset {:} (WMO ID: {:})'.format(dataset_id, row['wmo_id']))
            continue

        if dataset_ids or not clip_times:
            dataset_obs['osmc_gts_obs_count'] = obs.shape[0]
        else:
            dataset_obs['osmc_gts_obs_count'] = obs.loc[start_time:end_time].shape[0]

        obs_counts.append(dataset_obs)

    if not obs_counts:
        return 1

    df = pd.DataFrame(obs_counts).set_index('dataset_id')

    df['dataset_start_date'] = df.start_date.apply(lambda x: x.date)
    df['dataset_end_date'] = df.end_date.apply(lambda x: x.date)
    df.drop(columns=['start_date', 'end_date'], inplace=True)

    sys.stdout.write('DAC/OSMC Glider Activity Report: {:}\n'.format(
        datetime.datetime.utcnow().replace(tzinfo=pytz.UTC).strftime('%Y-%m-%d %H:%MZ')))

    if response == 'json':
        sys.stdout.write('{:}\n'.format(df.to_json(orient='records')))
    elif response == 'csv':
        sys.stdout.write('{:}\n'.format(df.to_csv()))
    else:
        sys.stdout.write(
            '{:}\n'.format(tabulate.tabulate(df, tablefmt=format, headers='keys')))
        logging.info('{:} data sets found'.format(df.shape[0]))

    return 0

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dataset_ids',
                            nargs='*',
                            help='One or more valid DAC data set IDs to search for')

    arg_parser.add_argument('-g', '--glider',
                            help='Return data sets with glider call signs starting with the specified string',
                            type=str)

    arg_parser.add_argument('--start_time',
                            dest='start_ts',
                            help='Search start time',
                            type=str)

    arg_parser.add_argument('--end_time',
                            dest='end_ts',
                            help='Search end time',
                            type=str)

    arg_parser.add_argument('--hours',
                            type=float,
                            help='Number of hours before now',
                            default=24)

    arg_parser.add_argument('-c', '--clip',
                            help='Clip profile and GTS obs counts to the specfied time search window',
                            action='store_true')

    arg_parser.add_argument('-n', '--north',
                            help='Maximum search latitude',
                            default=90.,
                            type=float)

    arg_parser.add_argument('-s', '--south',
                            help='Minimum search latitude',
                            default=-90.,
                            type=float)

    arg_parser.add_argument('-e', '--east',
                            help='Maximum search longitude',
                            default=180.,
                            type=float)

    arg_parser.add_argument('-w', '--west',
                            help='Minimum search longitude',
                            default=-180.,
                            type=float)

    arg_parser.add_argument('--gps',
                            help='Include lat/lon in GTS search query',
                            action='store_true')

    arg_parser.add_argument('--search_string',
                            help='Free format search string',
                            type=str)

    arg_parser.add_argument('-f', '--format',
                            help='Response format',
                            choices=['json', 'csv'] + tabulate.tabulate_formats,
                            default='csv')

    arg_parser.add_argument('-x', '--debug',
                            help='Debug mode. No operations performed',
                            action='store_true')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    # print(parsed_args)
    # sys.exit(13)

    sys.exit(main(parsed_args))
