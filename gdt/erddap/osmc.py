"""OSMC Glider GTS OSMCV4_DUO_PROFILES client"""
import pandas as pd
from erddapy import ERDDAP
import logging
import os
import io
import numpy as np
import requests
import urllib
from urllib.parse import urlsplit, urlunsplit, quote


class OsmcClient(object):
    # class DuoProfilesClient(object):

    def __init__(self, erddap_url=None):
        """OSMC ERDDAP OSMCV4_DUO_PROFILES dataset client for retrieving glider profile observations"""
        self._logger = logging.getLogger(os.path.basename(__file__))
        self._erddap_url = erddap_url or 'https://osmc.noaa.gov/erddap'
        self._protocol = 'tabledap'
        self._response_type = 'csv'
        self._timeout = 120

        self._datasets = ['OSMCV4_DUO_PROFILES',
                          'OSMC_30day',
                          'OSMC_flattened']

        self._dataset_id = self._datasets[0]

        self._client = ERDDAP(server=self._erddap_url, protocol=self._protocol, response=self._response_type)
        self._client.dataset_id = self._dataset_id

        self._last_request = None
        self._profiles = pd.DataFrame()
        self._obs = pd.DataFrame()

        self._dataset_series_fields = ['glider',
                                       'dataset_id',
                                       'wmo_id',
                                       'start_date',
                                       'end_date',
                                       'deployment_lat',
                                       'deployment_lon',
                                       'lat_min',
                                       'lat_max',
                                       'lon_min',
                                       'lon_max',
                                       'num_profiles',
                                       'days']
        self._profile_vars = ['time',
                              'platform_code',
                              'platform_type']

        self._profile_gps_vars = ['time',
                                  'platform_code',
                                  'platform_type',
                                  'latitude',
                                  'longitude']

        self._months = ['January',
                        'February',
                        'March',
                        'April',
                        'May',
                        'June',
                        'July',
                        'August',
                        'September',
                        'October',
                        'November',
                        'December']

    @property
    def e(self):
        """erddapy.ERDDAP client"""
        return self._client

    @property
    def profiles(self):
        return self._profiles

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, seconds=int):
        self._timeout = seconds

    @property
    def obs(self):
        return self._obs

    @property
    def datasets(self):
        return self._datasets

    @property
    def dataset_id(self):
        return self._client.dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        if dataset_id not in self._datasets:
            self._logger.error('Invalid dataset id: {:}'.format(dataset_id))
            return

        self._client.dataset_id = dataset_id

        self._logger.debug('OSMC dataset id: {:}'.format(self._client.dataset_id))

    @property
    def profiles_per_yyyymmdd(self):
        """Daily observation counts for all previously fetched observations"""
        if self._obs.empty:
            self._logger.warning('No GTS observations have been fetched')
            return pd.Series()

        profiles_by_yymmdd = self._obs.set_index('time').platform_code.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day]).count()
        profiles_by_yymmdd.name = 'num_profiles'
        profiles_by_yymmdd.index = pd.DatetimeIndex(
            pd.to_datetime(pd.DataFrame(list(profiles_by_yymmdd.index.values), columns=['year', 'month', 'day'])))

        return profiles_by_yymmdd

    @property
    def ymd_observations_calendar(self):

        if self._obs.empty:
            self._logger.warning('No GTS observations have been fetched')
            return pd.Series()

        calendar = self._obs.platform_code.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day]).count().unstack()

        # Fill in the missing (yyyy,mm) indices
        years = np.arange(calendar.index.levels[0].min(), calendar.index.levels[0].max() + 1)
        months = np.arange(calendar.index.levels[1].min(), calendar.index.levels[1].max() + 1)

        calendar.reindex(pd.MultiIndex.from_product([years, months]))

        for d in np.arange(1, 32):
            if d not in calendar.columns:
                calendar[d] = np.nan

        calendar.sort_index(axis=1, inplace=True)

        calendar.index.set_names(['year', 'month'], inplace=True)
        calendar.columns.set_names('day', inplace=True)

        return calendar

    @property
    def ym_observations_calendar(self):

        if self._obs.empty:
            self._logger.warning('No GTS observations have been fetched')
            return pd.Series()

        calendar = self._obs.platform_code.groupby(
            [lambda x: x.year, lambda x: x.month]).count().unstack()

        # Fill in the missing year indices
        years = np.arange(calendar.index.min(), calendar.index.max() + 1)
        calendar.reindex(pd.Index(years))

        for d in np.arange(1, 13):
            if d not in calendar.columns:
                calendar[d] = np.nan

        calendar.sort_index(axis=1, inplace=True)

        calendar.index.set_names('year', inplace=True)
        calendar.columns.set_names('month', inplace=True)

        return calendar

    def _send_obs_request(self, data_url):

        self._profiles = pd.DataFrame()

        try:
            r = requests.get(data_url, timeout=self._timeout)
            if r.status_code == 200:
                self._profiles = pd.read_csv(io.StringIO(r.text), skiprows=[1], parse_dates=True, index_col='time')
            elif r.status_code == 502:
                self._logger.warning('Retrying request: {:}'.format(data_url))

            return r.status_code

        except requests.Timeout as e:
            self._logger.warning('Request timeout for {:}'.format(data_url))
            return 408
        except Exception as e:
            self._logger.error('Fetch failed: {:} for {:}'.format(e, data_url))
            return 0

    def get_profiles_by_wmo_id(self, wmo_id, start_date, end_date, gps=True):

        constraints = {'platform_code=': wmo_id,
                       'time>=': start_date,
                       'time<=': end_date}

        obs_vars = self._profile_vars
        if gps:
            self._logger.debug('Including lat/lon in obs search')
            obs_vars = self._profile_gps_vars
        else:
            self._logger.debug('Omitting lat/lon from obs search')

        try:
            data_url = self._client.get_download_url(variables=obs_vars,
                                                     constraints=constraints,
                                                     distinct=True)
        except requests.exceptions.HTTPError as e:
            self._logger.warning(e)
            return

        self._last_request = data_url
        self._logger.debug('Request: {:}'.format(self._last_request))

        status = self._send_obs_request(data_url)
        self._logger.debug('Requst url: {:}'.format(data_url))
        if status == 502 or status == 408:
            logging.warning('Re-sending failed request: {:}'.format(data_url))
            status = self._send_obs_request(data_url)

        return self._profiles

    def get_dataset_profiles(self, datasets, gps=True):
        """Fetch the GTS profiles for the specified dataset.  Profiles are searched by wmo ID (platform_code) and
        dataset start_date and end_date.  Returns a pandas DataFrame"""

        self._logger.debug('Fetching GTS observations from {:}'.format(self._dataset_id))

        if isinstance(datasets, pd.Series):
            datasets = datasets.to_frame().T

        all_profiles = []
        for dataset_id, row in datasets.iterrows():

            if not row.wmo_id:
                self._logger.warning('Skipping GTS fetch for {:}: No wmo id'.format(dataset_id))
                continue

            self._logger.info('Fetching GTS obs for {:}'.format(dataset_id))
            profiles = self.get_profiles_by_wmo_id(row['wmo_id'],
                                                   row['start_date'],
                                                   row['end_date'],
                                                   gps=gps)

            self._logger.info('Found {:} GTS observations for {:}'.format(profiles.shape[0], dataset_id))

            if not profiles.empty:
                profiles['dataset_id'] = dataset_id
                all_profiles.append(profiles)

        if not all_profiles:
            return pd.DataFrame()

        self._obs = pd.concat(all_profiles)
        return self._obs

    @staticmethod
    def encode_url(data_url):
        """Percent encode special url characters."""
        url_pieces = list(urlsplit(data_url))
        url_pieces[3] = quote(url_pieces[3])

        return urlunsplit(url_pieces)

    def __repr__(self):
        return "<DuoProfilesClient(server='{:}', response='{:}', dataset_id={:})>".format(self._client.server,
                                                                                          self._client.response,
                                                                                          self._client.dataset_id)
