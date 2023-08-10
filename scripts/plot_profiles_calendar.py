#!/usr/bin/env python

import sys
import os
import argparse
import logging
import datetime
import tabulate
import pytz
from dateutil import parser
from gdt.erddap import GdacClient
from gdt.plotting.calendars import plot_calendar
import matplotlib.pyplot as plt
from pprint import pprint as pp


def main(args):
    """
    Search the IOOS Glider DAC for data sets matching the search criteria and plot the profiles calendar. Calendar is
    displayed as year, month and day and are restricted to the time search window.
    """

    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    dataset_ids = args.dataset_ids
    img_path = args.img_path
    title = args.title
    plot_type = args.plottype
    plot_all = args.all
    clobber = args.clobber
    hours = args.hours
    start_ts = args.start_ts
    end_ts = args.end_ts
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

    start_ts = start_time.strftime('%Y-%m-%dT%H:%M')
    end_ts = end_time.strftime('%Y-%m-%dT%H:%M')
    logging.info('Search start time: {:}'.format(start_ts))
    logging.info('Search end time  : {:}'.format(end_ts))

    params = {'min_time': start_ts,
              'max_time': end_ts,
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

    if plot_type == 'ymd':

        logging.info('Plotting year,month,day profiles calendar')

        # Timestamps for image names
        date0 = start_time.strftime('%Y%m%dT%H00Z')
        date1 = end_time.strftime('%Y%m%dT%H00Z')

        img_name = ''
        if img_path:
            img_name = os.path.join(img_path, 'ymd_profiles_{:}-{:}.png'.format(date0, date1))
            if os.path.isfile(img_name):
                if not clobber:
                    logging.warning('Image exists (Use -c to clobbber): {:}'.format(img_name))
                    return 1

        calendar = client.ymd_profiles_calendar

        if not plot_all:
            ym0 = (start_time.year, start_time.month)
            ym1 = (end_time.year, end_time.month)
            calendar = calendar.loc[ym0:ym1]

        # deployments calendar
        annot_kws = {'fontsize': 10}
        max_profiles = calendar.max().max()
        if max_profiles >= 2000:
            annot_kws['fontsize'] = 6
        elif max_profiles >= 1000:
            annot_kws['fontsize'] = 8

        ax = plot_calendar(calendar, annot_kws=annot_kws)

        logging.info('{:} total total profiles in the specified time window'.format(calendar.sum().sum()))

        ax.set_title('{:} Profiles ({:} Deployments): {:} - {:}'.format(title,
                                                               datasets.shape[0],
                                                               start_time.strftime('%b %d, %Y'),
                                                               end_time.strftime('%b %d, %Y')))
        if img_path:
            logging.info('Writing {:}'.format(img_name))
            plt.savefig(img_name, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    elif plot_type == 'ym':

        logging.info('Plotting year,month profiles calendar')

        # Timestamps for image names
        date0 = start_time.strftime('%Y%m%dT%H00Z')
        date1 = end_time.strftime('%Y%m%dT%H00Z')

        img_name = ''
        if img_path:
            img_name = os.path.join(img_path, 'ym_profiles_{:}-{:}.png'.format(date0, date1))
            if os.path.isfile(img_name):
                if not clobber:
                    logging.warning('Image exists (Use -c to clobbber): {:}'.format(img_name))
                    return 1

        calendar = client.ym_profiles_calendar

        if not plot_all:

            # Set cutoff year month tuples
            y0 = start_time.year
            y1 = end_time.year
            m0 = start_time.month
            m1 = end_time.month
            calendar = calendar.loc[y0:y1, m0:m1]

        # deployments calendar
        ax = plot_calendar(calendar)

        logging.info('{:} total total profiles in the specified time window'.format(calendar.sum().sum()))

        ax.set_title('{:} Profiles ({:} Deployments): {:} - {:}'.format(title,
                                                               datasets.shape[0],
                                                               start_time.strftime('%b %d, %Y'),
                                                               end_time.strftime('%b %d, %Y')))

        if img_path:
            logging.info('Writing {:}'.format(img_name))
            plt.savefig(img_name, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    return 0


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dataset_ids',
                            nargs='*',
                            help='One or more valid DAC data set IDs to search for')

    arg_parser.add_argument('-t','--plottype',
                            help='Type of calendar to plot',
                            choices=['ym', 'ymd'],
                            default='ymd')

    arg_parser.add_argument('-o', '--output_path',
                            dest='img_path',
                            type=str,
                            help='Image write destination (must exist). If not specified, the plot is displayed only.')

    arg_parser.add_argument('--title',
                            type=str,
                            default='',
                            help='Text to be prepended to the default figure title')

    arg_parser.add_argument('-a','--all',
                            help='Plot counts for all years, months, (and days), not just those in the specified time'
                                 'window',
                            action='store_true')

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

    arg_parser.add_argument('-c', '--clobber',
                            help='Clobber existing image',
                            action='store_true')

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
