import logging
from erddapy import ERDDAP
import pandas as pd
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import urllib
from urllib.parse import urlsplit, urlunsplit, quote
from requests.exceptions import HTTPError, ConnectionError
import requests
import io
import urllib3
from gdt.geo import latlon_to_geojson_track
from gdt.urls import end_points


class GdacClient(object):

    def __init__(self, erddap_url=None):
        """
        Create a connection to the ERDDAP server located at erddap_url used to search glider data sets.

        :param erddap_url: ERDDAP url
        """

        self._logger = logging.getLogger(os.path.basename(__file__))

        self._erddap_url = erddap_url or end_points['NGDAC_ERDDAP_URL'].url
        self._api_url = end_points['NGDAC_API_DEPLOYMENTS_URL'].url
        self._protocol = 'tabledap'
        self._response_type = 'csv'
        self._items_per_page = 1e6
        self._page = 1
        self._client = ERDDAP(server=self._erddap_url, protocol=self._protocol, response=self._response_type)
        self._timeout = (30, None)
        self._client.requests_kwargs['timeout'] = self._timeout
        self._last_request = None

        # DataFrame containing the allDatasets records
        self._erddap_datasets = pd.DataFrame()

        # DataFrame containing the results of ERDDAP advanced search (endpoints, etc.)
        self._datasets_info = pd.DataFrame()

        # DataFrame containing derived parameters (start/end dates, profile count, etc.). Results from a request url
        # for self._profile_variables. Merged with self._datasets_info to create self.datasets
        self._datasets_summaries = pd.DataFrame()

        # DataFrame indexed by date with dataset_id column names.  Cell value is the number of profiles found on that
        # day and nan if no profiles were found
        self._datasets_profiles = pd.DataFrame()

        # DataFrame indexed by date with dataset_id column names.  Cell value is set to 1 if the glider reported in that
        # day and nan if it did not
        self._datasets_days = pd.DataFrame()

        # DataFrame containing the daily averaged profile positions for each dataset_id
        self._daily_profile_positions = pd.DataFrame()

        self._profile_variables = ['time',
                                   'latitude',
                                   'longitude',
                                   'profile_id']

        self._calendar_types = ['datasets',
                                'days',
                                'profiles']

        self.fetch_erddap_datasets()

    @property
    def api_url(self):
        return self._api_url

    @property
    def datasets(self):
        """
        DataFrame resulting from the joining of self._datasets_summaries and self._datasets_info
        """
        if self._datasets_summaries.empty:
            return pd.DataFrame()

        return self._datasets_summaries.join(self._datasets_info)

    @property
    def erddap_datasets(self):
        """
        DataFrame containing all data sets resulting from the Advanced Search with no parameters. All data sets on the
        ERDDAP server are returned.

        :return: DataFrame
        """
        return self._erddap_datasets

    @property
    def daily_profile_positions(self):
        return self._daily_profile_positions

    @property
    def datasets_profiles(self):
        return self._datasets_profiles

    @property
    def datasets_days(self):
        return self._datasets_days

    @property
    def dataset_ids(self):
        if self._datasets_info.empty:
            # self._logger.warning('No data sets found')
            return []

        return list(self._datasets_info.index.values)

    @property
    def gliders(self):
        if self._datasets_summaries.empty:
            return []

        return list(self._datasets_summaries.glider.unique())

    @property
    def profiles_per_yyyymmdd(self):
        return self._datasets_profiles.sum(axis=1)

    @property
    def profiles_per_year(self):
        return self._datasets_profiles.sum(axis=1).groupby(lambda x: x.year).sum()

    @property
    def ymd_profiles_calendar(self):
        calendar = self.profiles_per_yyyymmdd.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum().unstack()

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
    def ym_profiles_calendar(self):
        calendar = self.profiles_per_yyyymmdd.groupby([lambda x: x.year, lambda x: x.month]).sum().unstack()

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

    @property
    def md_profiles_calendar(self):
        calendar = self.profiles_per_yyyymmdd.groupby([lambda x: x.month, lambda x: x.day]).sum().unstack()

        # Fill in the missing month indices
        calendar.reindex(pd.Index(np.arange(1, 13)))

        for d in np.arange(1, 32):
            if d not in calendar.columns:
                calendar[d] = np.nan

        calendar.sort_index(axis=1, inplace=True)

        calendar.index.set_names('month', inplace=True)
        calendar.columns.set_names('day', inplace=True)

        return calendar

    @property
    def glider_days_per_yyyymmdd(self):
        return self._datasets_days.sum(axis=1)

    @property
    def glider_days_per_year(self):
        return self._datasets_days.sum(axis=1).groupby(lambda x: x.year).sum()

    @property
    def ymd_glider_days_calendar(self):
        calendar = self._datasets_days.sum(axis=1).groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum().unstack()

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
    def ym_glider_days_calendar(self):
        calendar = self._datasets_days.sum(axis=1).groupby([lambda x: x.year, lambda x: x.month]).sum().unstack()

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

    @property
    def md_glider_days_calendar(self):
        calendar = self._datasets_days.sum(axis=1).groupby([lambda x: x.month, lambda x: x.day]).sum().unstack()

        # Fill in the missing month indices
        calendar.reindex(pd.Index(np.arange(1, 13)))

        for d in np.arange(1, 32):
            if d not in calendar.columns:
                calendar[d] = np.nan

        calendar.sort_index(axis=1, inplace=True)

        calendar.index.set_names('month', inplace=True)
        calendar.columns.set_names('day', inplace=True)

        return calendar

    @property
    def deployments_per_yyyymmdd(self):
        return self._datasets_days.sum(axis=1)

    @property
    def deployments_per_year(self):
        return self._datasets_days.groupby(lambda x: x.year).any().sum(axis=1)

    @property
    def ymd_deployments_calendar(self):
        calendar = self._datasets_days.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).any().sum(
            axis=1).unstack()

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
    def ym_deployments_calendar(self):
        calendar = self._datasets_days.groupby([lambda x: x.year, lambda x: x.month]).any().sum(axis=1).unstack()

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

    @property
    def md_deployments_calendar(self):
        calendar = self._datasets_days.groupby([lambda x: x.month, lambda x: x.day]).any().sum(axis=1).unstack()

        # Fill in the missing month indices
        calendar.reindex(pd.Index(np.arange(1, 13)))

        for d in np.arange(1, 32):
            if d not in calendar.columns:
                calendar[d] = np.nan

        calendar.sort_index(axis=1, inplace=True)

        calendar.index.set_names('month', inplace=True)
        calendar.columns.set_names('day', inplace=True)

        return calendar

    @property
    def yearly_stats(self):

        columns = [self.deployments_per_year, self.glider_days_per_year, self.profiles_per_year]
        totals = pd.DataFrame(columns).transpose().astype('i')
        totals.columns = ['deployments', 'glider days', 'profiles']
        totals.index.name = 'year'

        return totals

    @property
    def e(self):
        """erddapy.ERDDAP client"""
        return self._client

    @property
    def server(self):
        return self._client.server

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, seconds):
        self._timeout = seconds
        self._logger.info('Setting request timeout: {:}'.format(self._timeout))
        self._client.requests_kwargs['timeout'] = self._timeout

    @property
    def response_type(self):
        return self._client.response

    @property
    def last_request(self):
        return self._last_request

    def fetch_erddap_datasets(self):
        """
        ERDDAP Advanced search to return all available data sets.  The url is contained in self._erddap_url

        :return:
        self.erddap_datasets: Pandas DataFrame containing the result of the Advanced Search
        """

        try:

            self._logger.info('Fetching allDatasets: {:}'.format(self._erddap_url))
            url = self._client.get_download_url(dataset_id='allDatasets')
            self._last_request = url

            self._erddap_datasets = pd.read_csv(url, parse_dates=['minTime', 'maxTime'], skiprows=[1])

            # rename columns more friendly by replacing spaces with underscores and lower casing everything
            columns = {s: s.replace(' ', '_').lower() for s in self._erddap_datasets.columns}
            columns['datasetID'] = 'dataset_id'
            self._erddap_datasets.rename(columns=columns, inplace=True)

            # Use dataset_id as the index
            self._erddap_datasets.set_index('dataset_id', inplace=True)

            self._logger.info('{:} datasets available'.format(self._erddap_datasets.shape[0] - 1))

        except (requests.exceptions.HTTPError, urllib.error.URLError) as e:
            self._logger.error('Failed to fetch/parse ERDDAP server datasets info: {:} ({:})'.format(url, e))
            return pd.DataFrame()

    def get_glider_datasets(self, glider):
        """
        Return all found data sets for the specified glider name.

        :param glider: glider name
        :return: DataFrame
        """

        return self.datasets[self.datasets.glider == glider]

    def get_dataset_wmo_id(self, dataset_id):
        """
        Fetch the WMO ID, if assigned and valid, for the specified dataset_id

        :param dataset_id: valid ERDDAP dataset id
        :return: Valid WMO ID or empty string
        """

        wmo_id = None
        if dataset_id not in self._erddap_datasets.index:
            self._logger.warning('Cannot fetch WMO id for invalid dataset id {:}'.format(dataset_id))
            return wmo_id

        # WMO id
        self._logger.debug('Fetching {:} WMO ID'.format(dataset_id))
        wmo_id_url = self.e.get_download_url(dataset_id, variables=['wmo_id'], distinct=True)
        self._last_request = wmo_id_url
        try:
            self._logger.debug('wmo id GET: {:}'.format(self._last_request))
            r = requests.get(wmo_id_url, timeout=self._timeout, stream=True)
            if r.status_code != 200:
                self._logger.error('Request failed [reason={:}, code={:}]'.format(r.reason, r.status_code))
                self._logger.error('Request: {:}'.format(wmo_id_url))
                return None

            wmo_ids = pd.read_csv(io.StringIO(r.text), skiprows=[1])
            if wmo_ids.empty:
                self._logger.warning('No WMO ID found for {:}'.format(dataset_id))
            elif wmo_ids.shape[0] > 1:
                self._logger.warning('Multiple WMO IDs found for {:}'.format(dataset_id))
                for i, row in wmo_ids.iterrows():
                    self._logger.warning('{:} WMO id: {:}'.format(dataset_id, row.wmo_id))
            else:
                wmo_id = wmo_ids.iloc[0].values.astype('int').astype('str')[0]
                self._logger.info('WMO id for data set {:} is {:} '.format(dataset_id, wmo_id))

        except requests.exceptions.RequestException as e:
            self._logger.error('Failed to fetch WMO ID for {:}: reason={:}'.format(dataset_id, e))
            return None
        except ValueError as e:
            self._logger.warning('Invalid WMO id {:} for {:}: {:}'.format(wmo_id, dataset_id, e))
            return None

        return wmo_id

    def search_dataset_ids(self, text_blob, start=0, end=None):
        """
        Search for all data set ids that contain the text_blob in the name

        :param text_blob: text string to search for
        :param start: start search at this character index. Default is the start of the dataset id
        :param end: end the search at this character index. Default is the end of the dataset_id
        :return: list of found dataset ids
        """

        return self.erddap_datasets[
            self.erddap_datasets.index.str.find(text_blob, start=start, end=end) != -1].index.to_list()

    def search_datasets(self, params=None, dataset_ids=None, include_delayed_mode=False):
        """Search the ERDDAP server for glider deployment datasets using ERDDAP's Advanced Search. Additional requests
        for profile information are made for each found data set and merged with the search results.

        Results are stored as a pandas DataFrame in:

        self.datasets

        Equivalent to ERDDAP's Advanced Search.  Searches can be performed by free text, bounding box, time bounds, etc.
        See the erddapy documentation for valid kwargs"""

        if params is None:
            params = {}

        glider_regex = re.compile(r'^(.*)-\d{8}T\d{4}')

        summary_columns = ['glider',
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

        if dataset_ids and not isinstance(dataset_ids, list):
            dataset_ids = [dataset_ids]

        # Get the search url for the specified parameters
        url = self._client.get_search_url(items_per_page=self._items_per_page, **params)
        self._logger.debug(url)
        self._last_request = url
        self._logger.debug('Advanced Search GET: {:}'.format(self._last_request))

        try:
            self._datasets_info = pd.read_csv(url)
            # Drop the allDatasets row
            self._datasets_info.drop(self._datasets_info[self._datasets_info['Dataset ID'] == 'allDatasets'].index,
                                     inplace=True)

            # rename columns more friendly
            columns = {s: s.replace(' ', '_').lower() for s in self._datasets_info.columns}
            self._datasets_info.rename(columns=columns, inplace=True)

            if not include_delayed_mode:
                self._logger.info('Excluding delayed mode datasets')
                self._datasets_info = self._datasets_info[~self._datasets_info.dataset_id.str.endswith('delayed')]
            else:
                self._logger.info('Including delayed mode datasets')

            # Reset the index to start and 0
            self._datasets_info = self._datasets_info.set_index('dataset_id').drop(['griddap', 'wms'], axis=1).replace(
                ['None', None], False)

            # Eliminate rows that are not from the dataset_ids specified (if specified)
            if dataset_ids:
                drop_dataset_ids = [did for did, row in self._datasets_info.iterrows() if did not in dataset_ids]
                if drop_dataset_ids:
                    self._datasets_info = self._datasets_info.drop(index=drop_dataset_ids)

        except urllib.error.HTTPError as e:
            self._logger.warning('code={:}: query produced no matching results. (nRows = 0)'.format(e.code))
        except (requests.exceptions.HTTPError, urllib.error.URLError) as e:
            self._logger.error('Failed to fetch/parse ERDDAP server datasets info: {:} ({:})'.format(url, e))
            return

        self._logger.info('{:} ERDDAP datasets found with specified criteria'.format(self._datasets_info.shape[0]))
        if self._datasets_info.empty:
            return

        # Iterate through each data set (except for allDatasets) and grab the info page
        datasets = []
        daily_profiles = []
        datasets_days = []
        avg_profile_pos = []
        for dataset_id, row in self._datasets_info.iterrows():

            # Fetch the WMO id for this dataset, if there is one
            wmo_id = self.get_dataset_wmo_id(dataset_id)
            if not wmo_id:
                wmo_id = ''

            # Harvest all of the profiles contained in the data set (contained in self._profile_variables).
            self._logger.debug('Fetching {:} profiles'.format(dataset_id))

            try:
                # Get the download url for profiles
                self._logger.debug('Creating {:} profiles download url'.format(dataset_id))
                data_url = self._client.get_download_url(dataset_id=dataset_id,
                                                         variables=self._profile_variables)
                self._last_request = data_url
            except (ConnectionError, ConnectionRefusedError, urllib3.exceptions.MaxRetryError,
                    requests.exceptions.HTTPError) as e:
                self._logger.error('{:} fetch failed: {:}'.format(dataset_id, e))
                continue

            # Fetch the profiles into a pandas dataframe
            try:
                self._logger.debug('Profiles GET: {:}'.format(self._last_request))

                r = requests.get(data_url, timeout=self._timeout, stream=True)
                if r.status_code != 200:
                    self._logger.error('Request failed (reason={:}, code={:})'.format(r.reason, r.status_code))
                    self._logger.error('Request: {:}'.format(data_url))
                    continue

                profiles = pd.read_csv(io.StringIO(r.text), skiprows=[1], index_col='time', parse_dates=True,
                                       na_values=['none', 'None']).sort_index()

                self._logger.debug('Found {:} profiles for data set {:} ({:0.1f} seconds)'.format(profiles.shape[0],
                                                                                                 dataset_id,
                                                                                                 r.elapsed.total_seconds()))

            except requests.exceptions.RequestException as e:
                self._logger.error('Failed to fetch {:} profiles: {:}'.format(dataset_id, e))
                continue

            # Group the profile by date, average the latitude and longitude and get a daily count
            profile_stats = profiles.groupby(lambda x: x.date).agg(
                {'latitude': 'mean', 'longitude': 'mean', 'profile_id': 'size'}).rename(
                columns={'profile_id': 'num_profiles'})
            profile_stats['dataset_id'] = dataset_id

            s = profile_stats.num_profiles
            s.name = dataset_id

            avg_profile_pos.append(profile_stats[['dataset_id', 'latitude', 'longitude']])

            daily_profiles.append(s)

            # Create the deployment date range
            d_index = pd.date_range(s.index.min(), s.index.max())
            deployment_days = pd.Series([1 for x in d_index], index=d_index, name=dataset_id)
            datasets_days.append(deployment_days)

            glider_match = glider_regex.match(dataset_id)
            glider = glider_match.groups()[0]

            # First profile time
            dt0 = profiles.index.min()
            # Last profile time
            dt1 = profiles.index.max()
            # Deployment length in days
            days = ceil((dt1 - dt0).total_seconds() / 86400)

            dataset_summary = [glider,
                               dataset_id,
                               str(wmo_id),
                               dt0,
                               dt1,
                               profiles.iloc[0]['latitude'],
                               profiles.iloc[0]['longitude'],
                               profiles.latitude.min(),
                               profiles.latitude.max(),
                               profiles.longitude.min(),
                               profiles.longitude.max(),
                               profiles.shape[0],
                               days
                               ]

            datasets.append(dataset_summary)

        if not datasets:
            self._logger.warning('No datasets returned from search')
            return

        self._datasets_summaries = pd.DataFrame(datasets, columns=summary_columns).set_index('dataset_id').replace(
            ['None', None], False)

        # Create and store the DataFrame containing a 1 on each day the glider was deployed, 0 otherwise
        self._datasets_days = pd.concat(datasets_days, axis=1).sort_index()

        # Create and store the DataFrame containing the number of profiles on each day for each deployment
        self._datasets_profiles = pd.concat(daily_profiles, axis=1).sort_index()
        self._datasets_profiles.index = pd.to_datetime(self._datasets_profiles.index)

        self._daily_profile_positions = pd.concat(avg_profile_pos, axis=0).reset_index().rename(
            columns={'index': 'date'})

        return

    def check_dataset_exists(self, dataset_id):
        """
        See if the specified dataset_id exists on the ERDDAP server.
        :param dataset_id: dataset_id
        :return: True if exists, False otherwise
        """

        if dataset_id not in self._erddap_datasets.index:
            self._logger.warning('Dataset id {:} not found on {:}'.format(dataset_id, self._client.server))
            return False

        return True

    def get_dataset_profiles(self, dataset_id):
        """
        Fetch all profiles (time, latitude, longitude, profile_id) for the specified dataset.  Profiles are sorted
        by ascending time
        """

        if dataset_id not in self._erddap_datasets.index:
            self._logger.warning('Dataset id {:} not found on {:}'.format(dataset_id, self._client.server))
            return pd.DataFrame()

        url = self._client.get_download_url(dataset_id=dataset_id, variables=self._profile_variables)
        self._last_request = url

        try:
            return pd.read_csv(url, parse_dates=True, skiprows=[1], index_col='time').sort_index()
        except urllib.error.HTTPError as e:
            self._logger.error('Request: {:}'.format(url))
            return pd.DataFrame()

    def get_dataset_time_series(self, dataset_id, variables, min_time=None, max_time=None):
        """Fetch the variables time-series for the specified dataset_id.  A time window can be specified using min_time
        and max_time, which must be ISO-8601 formatted date strings (i.e.: 'YYYY-mm-ddTHH:MM')

        Parameters
        dataset_id: valid dataset id from self.datasets
        variables: list of one or more valid variables in the dataset

        Options
        min_time: minimum time value formatted as 'YYYY-mm-ddTHH:MM[:SS]'
        max_time: maximum time value formatted as 'YYYY-mm-ddTHH:mm[:SS]'
        """
        if dataset_id not in self._erddap_datasets.index:
            self._logger.warning('Dataset id {:} not found on {:}'.format(dataset_id, self._client.server))
            return

        if not isinstance(variables, list):
            variables = [variables]

        # Make sure to include precise_time and depth in the request
        all_variables = ['precise_time', 'depth'] + variables
        variables = set(all_variables)

        constraints = {}
        if min_time:
            constraints['precise_time>='] = min_time
        if max_time:
            constraints['precise_time<='] = max_time

        # Not sure why, but pd.read_csv doesn't like percent UNENCODED urls on data requests, so percent escape special
        # characters prior to sending the data request.
        data_url = self.encode_url(
            self._client.get_download_url(dataset_id=dataset_id, variables=variables, constraints=constraints))

        self._last_request = data_url

        data = pd.DataFrame([])
        try:
            self._logger.debug('wmo id GET: {:}'.format(self._last_request))
            r = requests.get(data_url, timeout=self._timeout, stream=True)
            if r.status_code != 200:
                self._logger.error('Request failed [reason={:}, code={:}]'.format(r.reason, r.status_code))
                self._logger.error('Request: {:}'.format(data_url))
                return None

            return pd.read_csv(io.StringIO(r.text), skiprows=[1], parse_datas=True).set_index(
                'precise_time').sort_index()

        except Exception as e:
            self._logger.error('Failed to fetch {:} time-series: {:}'.format(dataset_id, e))
            return data

    def _get_dataset_description(self, dataset_id:str):

        info_url = self._client.get_info_url(dataset_id)

        self._last_request = info_url

        response = pd.read_csv(info_url)
        # Rename columns for friendlier access
        response = response.rename(columns={r: r.replace(' ', '_').lower() for r in response.columns})

        # Get the variable names
        var_names = response[response.row_type == 'variable']['variable_name'].values.tolist()

        cdl = {'NC_GLOBAL': {},
               'variables': {}
               }

        # NC_GLOBALs
        for i, row in response[response.variable_name == 'NC_GLOBAL'].iterrows():

            cdl['NC_GLOBAL'][row.attribute_name] = row['value']

        for var_name in var_names:

            props = response[response.variable_name == var_name]

            if props.empty:
                self._logger.warning('No description found for variable {:}'.format(var_name))
                continue

            desc = {'nc_var_name': var_name,
                    'type': '',
                    'attrs': {}
                    }

            # Set the dtype
            desc['type'] = props[props.row_type == 'variable']['data_type'].iloc[0].lower()

            for i, row in props[props.row_type == 'attribute'].iterrows():

                desc['attrs'][row.attribute_name] = row.value

            cdl['variables'][var_name] = desc

        return cdl

    def get_dataset_variables(self, dataset_id:str):
        """Get variable names for the specified dataset_id"""

        dataset_variables = []

        if not self.check_dataset_exists(dataset_id):
            return dataset_variables

        cdl = self._get_dataset_description(dataset_id)

        return list(cdl['variables'])

    def get_dataset_cdl(self, dataset_id):
        """Get the full data set global variables and variable description for the specified dataset_id"""

        return self._get_dataset_description(dataset_id)


    def plot_yearly_totals(self, totals_type=None, palette='Blues_d', **kwargs):
        """Bar chart plot of deployments, glider days and profiles, grouped by year.  The numbers are from those data
        sets returned by self.search_datasets.

        If specified, totals_type must be one of the following:

            'deployments'
            'glider days'
            'profiles'

        to plot ONLY that type.
        """
        totals = self.yearly_stats.reset_index()

        if totals_type and totals_type not in totals.columns:
            self._logger.error('Invalid category specified: {:}'.format(totals_type))
            return

        if not totals_type:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
            sns.barplot(x='year', y='deployments', ax=ax1, data=totals, palette=palette, **kwargs)
            sns.barplot(x='year', y='glider days', ax=ax2, data=totals, palette=palette, **kwargs)
            sns.barplot(x='year', y='profiles', ax=ax3, data=totals, palette=palette, **kwargs)

            ax2.set_xlabel('')
            ax1.set_xlabel('')

            ax1.set_title('U.S. IOOS Glider Data Assembly Center')

            return fig, ax1, ax2, ax3

        else:
            ax = sns.barplot(x='year', y=totals_type, data=totals, palette=palette, **kwargs)
            ax.set_title('U.S. IOOS Glider Data Assembly Center')

            return (ax.figure, ax)

    def get_dataset_track_geojson(self, dataset_id, points=True, precision='0.001'):

        if dataset_id not in self._erddap_datasets.index:
            self._logger.warning('Dataset id {:} not found on {:}'.format(dataset_id, self._client.server))
            return {}

        profiles = self.get_dataset_profiles(dataset_id)
        if profiles.empty:
            self._logger.warning('No profiles found for dataset ID: {:}'.format(dataset_id))
            return {}

        return latlon_to_geojson_track(profiles.latitude,
                                       profiles.longitude,
                                       profiles.index,
                                       include_points=points,
                                       precision=precision)

    def get_dataset_metadata(self, dataset_id):
        """
        Fetch the ERDDAP data set meta data (attributes, variables, etc.) for the specified dataset_id

        :param dataset_id: dataset_id
        :return:  DataFrame
        """

        try:
            info_url = self._client.get_info_url(dataset_id)
            self._last_request = info_url
            return pd.read_csv(info_url)
        except (ConnectionError, ConnectionRefusedError, urllib3.exceptions.MaxRetryError,
                requests.exceptions.HTTPError) as e:
            self._logger.error(e)
            return pd.DataFrame([])


    @staticmethod
    def encode_url(data_url):
        """Percent encode special url characters."""
        url_pieces = list(urlsplit(data_url))
        url_pieces[3] = quote(url_pieces[3])

        return urlunsplit(url_pieces)

    def __repr__(self):
        return "<GdacClient(server='{:}', response_type='{:}', num_datasets={:})>".format(self._client.server,
                                                                                          self._response_type,
                                                                                          len(self._datasets_info))
