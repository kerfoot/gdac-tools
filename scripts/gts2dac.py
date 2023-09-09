#!/usr/bin/env python

import sys
import argparse
import logging
import tabulate
from gdt.erddap import GdacClient
from gdt.erddap.osmc import OsmcClient
from gdt.urls import end_points
from gdt.gts import match_dac_profiles_to_gts_obs


def main(args):
    """Match available DAC profiles to released GTS observations for the specified dataset ID.

    Glider GTS observations are timestamped to the minute (YYYY-mm-ddTHH:MM), while DAC profiles are timestamped to
    the second (YYYY-mm-ddTHH:MM:SS). The timestamp for each DAC profile is truncated to the minute an compared to all
    GTS observation timestamps. A DAC profile is deemed to be released as a GTS observation if the truncated timestamp
    is found in the list of harvested GTS observations.
    """

    # Set up logger
    log_level = getattr(logging, args.loglevel.upper())
    log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    dataset_id = args.dataset_id
    osmc_dataset_id = args.osmc_dataset
    response = args.format
    include_gps = True
    verbose = args.verbose

    if len(dataset_id) > 1:
        logging.error('Only one valid dataset ID may be specified at a time')
        return 1

    client = GdacClient()
    if client.erddap_datasets.empty:
        logging.info('ERDDAP server contains no data sets: {:}'.format(client.server))
        return 1

    # Search the ERDDAP server
    client.search_datasets(dataset_ids=dataset_id)

    if client.datasets.empty:
        logging.warning('Dataset not found: {:}'.format(dataset_id))
        return 1

    dataset_id = client.datasets.iloc[0].name
    logging.info('Fetching DAC profiles: {:}'.format(dataset_id))
    dac_profiles = client.get_dataset_profiles(dataset_id)
    # Drop invalid profiles
    dac_profiles = dac_profiles.dropna()
    logging.info('Found {:} DAC profiles for {:}'.format(dac_profiles.shape[0], dataset_id))

    if dac_profiles.empty:
        return 1

    logging.info('Creating OSMC ERDDAP connection: {:}'.format(end_points['OSMC_ERDDAP_URL'].url))
    osmc_client = OsmcClient(erddap_url=end_points['OSMC_ERDDAP_URL'].url)
    if osmc_dataset_id not in osmc_client.datasets:
        logging.error('Dataset {:} not found at {:}'.format(osmc_dataset_id, osmc_client.e.server))
        return 1

    logging.info('OSMC ERDDAP dataset: {:}'.format(osmc_client.dataset_id))

    if include_gps:
        logging.info('Including GPS in GTS observation searches')
    else:
        logging.info('Omitting GPS in GTS observation searches')

    # Fetch available observations for the data set
    osmc_profiles = osmc_client.get_dataset_profiles(client.datasets, gps=include_gps)
    orig_num_obs = osmc_profiles.shape[0]
    logging.info('Found {:} GTS observations for {:}'.format(orig_num_obs, dataset_id))

    # Remove "duplicate" obs by:
    # 1. reset index to create 'time' column
    # 2. Round the latitude and longitude columns to 4 decimal places
    # 3. Drop duplicates, which is defined as having the same time,latitude,longitude
    # 4. Set the index of the resulting data frame to 'time'
    osmc_profiles = osmc_profiles.reset_index().round({'latitude': 4, 'longitude': 4}).drop_duplicates(
        subset=['time', 'latitude', 'longitude']).set_index('time')
    logging.info('Removed {:} duplicate observations'.format(orig_num_obs - osmc_profiles.shape[0]))

    aligned_profiles = match_dac_profiles_to_gts_obs(dac_profiles, osmc_profiles)
    if not aligned_profiles.empty:
        aligned_profiles = aligned_profiles.rename(columns={'profile_id': 'dac_profile_id', 'platform_code': 'wmo_id'})
        # Format lat/lon precision
        aligned_profiles = aligned_profiles.round(
            {'dac_latitude': 4, 'dac_longitude': 4, 'gts_latitude': 4, 'gts_longitude': 4})
        # Convert dac_profile_id to string for display purposes
        logging.info('Integerfying dac_profile_id')
        aligned_profiles.dac_profile_id = aligned_profiles.dac_profile_id.astype(int)

    if response == 'json':
        sys.stdout.write('{:}\n'.format(aligned_profiles.reset_index().to_json(orient='records',
                                                                               lines=True,
                                                                               date_unit='s',
                                                                               date_format='iso',
                                                                               indent=4)))
    elif response == 'csv':
        sys.stdout.write('{:}\n'.format(aligned_profiles.to_csv(date_format='%Y-%m-%dT%H:%M:%SZ')))
    else:
        if verbose:
            columns = ['dac_timestamp',
                       'gts_count',
                       'platform_type',
                       'wmo_id',
                       'gts_latitude',
                       'dac_latitude',
                       'gts_longitude',
                       'dac_longitude',
                       'dac_profile_id',
                       'dataset_id']
        else:
            columns = ['dac_timestamp',
                       'gts_count',
                       'platform_type',
                       'wmo_id',
                       'dac_profile_id',
                       'dataset_id']
        sys.stdout.write(
            '{:}\n'.format(tabulate.tabulate(aligned_profiles[columns], tablefmt=response, headers='keys')))

    logging.info('Released DAC profiles: {:}'.format(dac_profiles.shape[0]))
    logging.info('Released OSMC GTS obs: {:}'.format(osmc_profiles.shape[0]))

    return 0


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('dataset_id',
                            nargs='+',
                            help='A single, valid DAC data set ID')

    arg_parser.add_argument('-v', '--verbose',
                            dest='verbose',
                            help='Display all columns',
                            action='store_true')

    _ = OsmcClient()

    arg_parser.add_argument('-d', '--osmc_dataset',
                            help='OSMC ERDDAP data set to query',
                            choices=_.datasets,
                            default=_.dataset_id)

    arg_parser.add_argument('-f', '--format',
                            help='Response format',
                            choices=['json', 'csv'] + tabulate.tabulate_formats,
                            default='csv')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    #    print(parsed_args)
    #    sys.exit(13)

    sys.exit(main(parsed_args))
