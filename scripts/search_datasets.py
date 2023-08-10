#!/usr/bin/env python

import sys
import argparse
import logging
import datetime
import tabulate
import pytz
from dateutil import parser
from gdt.erddap import GdacClient
from pprint import pprint as pp


def main(args):
    """Search the IOOS Glider DAC and return the dataset ids for all datasets which have updated within the last 24
    hours"""

    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    dataset_ids = args.dataset_ids
    hours = args.hours
    start_ts = args.start_ts
    end_ts = args.end_ts
    response = args.format
    long_format = args.long_format
    debug = args.debug
    north = args.north
    south = args.south
    east = args.east
    west = args.west
    search_string = args.search_string or ''
    variable_names = args.variables or ''
    glider = args.glider

    client = GdacClient()
    if client.erddap_datasets.empty:
        logging.info('ERDDAP server contains no data sets: {:}'.format(client.server))
        return 1

    # Use the min and max times from allDatasets
    start_time = client.erddap_datasets.mintime.min()
    end_time = client.erddap_datasets.maxtime.max()

    params = {}
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
              'variableName': variable_names
              }

    # Search the ERDDAP server
    client.search_datasets(params=params, dataset_ids=dataset_ids)

    if client.datasets.empty:
        logging.warning('No datasets found matching the search criteria')
        return 1

    datasets = client.datasets

    if glider:
        logging.info('Filtering search results for gliders starting with {:}'.format(glider))
        datasets = datasets[datasets.glider.str.startswith(glider)]

    logging.info('Found {:} data sets matching search criteria'.format(datasets.shape[0]))

    # Calculate and add the latency
    tdelta = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC) - datasets.end_date
    datasets['latency_minutes'] = tdelta.dt.total_seconds()/60
    datasets['latency_minutes'] = datasets['latency_minutes'].map('{:0.0f}'.format)

    print_columns = ['glider',
                     'wmo_id',
                     'start_date',
                     'end_date',
                     'lat_min',
                     'lat_max',
                     'lon_min',
                     'lon_max',
                     'num_profiles',
                     'days',
                     'latency_minutes']

    if not long_format:
        datasets = datasets[print_columns]

    if response == 'json':
        sys.stdout.write('{:}\n'.format(datasets.to_json(orient='records')))
    elif response == 'csv':
        sys.stdout.write('{:}\n'.format(datasets.to_csv()))
    else:
        sys.stdout.write(
            '{:}\n'.format(tabulate.tabulate(datasets[print_columns], tablefmt=format, headers='keys')))
        logging.info('{:} data sets found'.format(datasets.shape[0]))

    return 0


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dataset_ids',
                            nargs='*',
                            help='One or more valid DAC data set IDs to search for')

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
                            help='Number of hours before now. Set to <= 0 to disable',
                            default=24)

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

    arg_parser.add_argument('-v', '--variables',
                            nargs='*',
                            help='Data set must contain one or more of the specified variables',
                            type=str)

    arg_parser.add_argument('-g', '--glider',
                            help='Return data sets with glider call signs starting with the specified string',
                            type=str)

    arg_parser.add_argument('--search_string',
                            help='Free format search string',
                            type=str)

    arg_parser.add_argument('--long',
                            dest='long_format',
                            action='store_true',
                            help='Long format. Include ERDDAP urls')

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

#    print(parsed_args)
#    sys.exit(13)

    sys.exit(main(parsed_args))
