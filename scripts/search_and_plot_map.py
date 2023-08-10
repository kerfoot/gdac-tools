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
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def main(args):
    """Search the IOOS Glider DAC for all datasets which have updated within the last 24 hours and plot the resulting
    tracks on a map"""

    # Set up logger
    log_level = getattr(logging, 'INFO')
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    dataset_ids = args.dataset_ids
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
    img_name = args.img_name
    clobber = args.clobber
    valid_image_types = ['png',
                         'jpg',
                         'pdf',
                         'svg']

    # Plotting args
    central_longitude = args.central_longitude
    projection = args.projection
    global_map = args.global_map
    clamp_bounds = args.clamp
    edge_color = "black"
    land_color = "tan"
    ocean_color = cfeature.COLORS['water']  # cfeature.COLORS['water'] is the standard
    marker = 'None'
    marker_size = 1.0
    linestyle = '-'

    if img_name:
        (img_path, iname) = os.path.split(img_name)
        if not os.path.isdir(img_path):
            logging.error('Specified image destination directory does not exist: {:}'.format(img_path))
            return 1

        tokens = iname.split('.')
        if len(tokens) < 2:
            logging.info('No image type specified. Valid image types are: {:}'.format(valid_image_types))
            return 1

        if tokens[-1] not in valid_image_types:
            logging.info('Invalid image type specified: {:}'.format(tokens[-1]))
            logging.info('Valid image types are: {:}'.format(valid_image_types))
            return 1

        if os.path.isfile(img_name):
            if not clobber:
                logging.warning('Image exists (Use -c to clobbber): {:}'.format(img_name))
                return 1

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
    if datasets.empty:
        return 0

    if debug:
        logging.info('Debug (-x). Skipping map creation')
        return 0

    # Calculate the central longitude if not specified
    if not central_longitude:
        central_longitude = client.datasets[['lon_min', 'lon_max']].mean().mean()

    # Create the extents
    bbox = [west,
            east,
            south,
            north]

    if clamp_bounds:
        bbox = [client.datasets.lon_min.min(),
                client.datasets.lon_max.max(),
                client.datasets.lat_min.min(),
                client.datasets.lat_max.max()]

    kws = {'projection': getattr(ccrs, projection)(central_longitude=central_longitude)}
    map_fig, map_ax = plt.subplots(figsize=(11, 8), subplot_kw=kws)

    # Set extent
    if global_map:
        logging.info('Setting global extent')
        map_ax.set_global()
    elif clamp_bounds:
        logging.info('Setting extent to bounds resulting from found data sets')
        map_ax.set_extent(bbox)
    else:
        logging.info('Setting extent to search bounds')
        map_ax.set_extent(bbox)

    map_ax.set_facecolor(ocean_color)  # way faster than adding the ocean feature above

    # Land
    lakes = cfeature.NaturalEarthFeature(category='physical',
                                         name='land',
                                         scale='10m',
                                         edgecolor=edge_color,
                                         facecolor=land_color)
    map_ax.add_feature(lakes,
                       zorder=0)

    # Lakes/Rivers
    land = cfeature.NaturalEarthFeature(category='physical',
                                        name='lakes',
                                        scale='110m',
                                        edgecolor=edge_color,
                                        facecolor=ocean_color)
    map_ax.add_feature(land,
                       zorder=1)

    cbar = mpl.cm.get_cmap('rainbow', client.datasets.shape[0])
    i = 0
    for dataset_id, row in client.datasets.iterrows():

        gps = client.daily_profile_positions[client.daily_profile_positions.dataset_id == dataset_id]
        if gps.empty:
            logging.warning('No GPS track found for {:}'.format(dataset_id))

        track = gps.sort_values('date', ascending=True)

        # Plot the track
        map_ax.plot(track.longitude, track.latitude, marker=marker, markersize=marker_size, linestyle=linestyle,
                    color=cbar(i),
                    transform=ccrs.PlateCarree())

        i += 1

    if img_name:
        logging.info('Writing image: {:}'.format(img_name))
        plt.savefig(img_name, dpi=300)
    else:
        logging.info('Displaying image')
        plt.show()

    return 0


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dataset_ids',
                            nargs='*',
                            help='One or more valid DAC data set IDs to search for')

    arg_parser.add_argument('-o', '--image_name',
                            dest='img_name',
                            help='Write image to the specified filename. If not specified, image is displayed. The '
                                 'extension specifies the image type',
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

    arg_parser.add_argument('--global',
                            dest='global_map',
                            help='Set map bounds to global',
                            action='store_true')

    arg_parser.add_argument('-p', '--projection',
                            help='Set map projection',
                            choices=['PlateCarree', 'Mollweide', 'Robinson'],
                            default='PlateCarree',
                            type=str)

    arg_parser.add_argument('--central_longitude',
                            help='Specify longitude for map center',
                            type=float)

    arg_parser.add_argument('--clamp',
                            help='Set map bounds to bounding box defined by data set search results',
                            action='store_true')

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
