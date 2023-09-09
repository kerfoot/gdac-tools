import logging
import pandas as pd
import pytz

logging.getLogger(__file__)


def match_gts_obs_to_dac_profiles(osmc_profiles, dac_profiles):
    """Match all glider DAC profiles to their corresponding GTS observation in the GTS OSMC profiles. The number of GTS
    observations for each profile is counted.

    Glider GTS observations are timestamped to the minute (YYYY-mm-ddTHH:MM), while DAC profiles are timestamped to
    the second (YYYY-mm-ddTHH:MM:SS). The timestamp for each DAC profile is truncated to the minute an compared to all
    GTS observation timestamps. A DAC profile is deemed to be released as a GTS observation if the truncated timestamp
    is found in the list of harvested GTS observations.
    """

    if osmc_profiles.empty:
        return pd.DataFrame()

    if dac_profiles.empty:
        return pd.DataFrame()

    aligned_profiles = dac_profiles.copy(deep=True).dropna()

    # Initialize a gts_count column and set to 0
    aligned_profiles['wmo_id'] = ''
    aligned_profiles['gts_count'] = 0
    aligned_profiles['platform_type'] = ''
    aligned_profiles['gts_timestamp'] = pd.NaT
    aligned_profiles['gts_latitude'] = pd.NA
    aligned_profiles['gts_longitude'] = pd.NA

    for dac_ts, row in aligned_profiles.iterrows():
        gts_dt = pd.to_datetime(dac_ts.strftime('%Y-%m-%d %H:%M:00')).replace(tzinfo=pytz.UTC)

        if gts_dt not in osmc_profiles.index:
            continue

        gts_obs = osmc_profiles[osmc_profiles.index == gts_dt]
        aligned_profiles.loc[dac_ts, 'gts_count'] = gts_obs.shape[0]
        aligned_profiles.loc[dac_ts, 'platform_type'] = ','.join(gts_obs.platform_type.unique().tolist())
        aligned_profiles.loc[dac_ts, 'gts_timestamp'] = gts_dt
        aligned_profiles.loc[dac_ts, 'gts_latitude'] = gts_obs.latitude.values[0]
        aligned_profiles.loc[dac_ts, 'gts_longitude'] = gts_obs.longitude.values[0]
        aligned_profiles.loc[dac_ts, 'wmo_id'] = '{:}'.format(gts_obs.platform_code.unique()[0])

    aligned_profiles.index.name = 'dac_timestamp'

    return aligned_profiles


def match_dac_profiles_to_gts_obs(dac_profiles, osmc_profiles):
    """Match all glider GTS OSMC observations to their corresponding DAC profile from the IOOS Glider DAC. The number of
    GTS observations for each profile is counted.

    Glider GTS observations are timestamped to the minute (YYYY-mm-ddTHH:MM), while DAC profiles are timestamped to
    the second (YYYY-mm-ddTHH:MM:SS). The timestamp for each DAC profile is truncated to the minute an compared to all
    GTS observation timestamps. A DAC profile is deemed to be released as a GTS observation if the truncated timestamp
    is found in the list of harvested GTS observations.
    """

    if osmc_profiles.empty:
        return pd.DataFrame()

    if dac_profiles.empty:
        return pd.DataFrame()

    aligned_profiles = osmc_profiles.copy(deep=True).dropna()

    aligned_profiles['dac_timestamp'] = pd.NaT
    aligned_profiles['dac_latitude'] = pd.NA
    aligned_profiles['dac_longitude'] = pd.NA
    aligned_profiles['dac_profile_id'] = pd.NA
    aligned_profiles['gts_count'] = 0

    # Drop NaT timestamps
    dac_profiles = dac_profiles.dropna()
    # Reindex using the original index truncated to minutes (seconds=00)
    dac_timestamps = pd.to_datetime(dac_profiles.index.strftime('%Y-%m-%dT%H:%M:00Z')).to_list()

    for gts_ts, row in aligned_profiles.iterrows():

        gts_obs_count = aligned_profiles[aligned_profiles.index == gts_ts].shape[0]
        aligned_profiles.loc[gts_ts, 'gts_count'] = gts_obs_count

        if gts_ts not in dac_timestamps:
            continue

        aligned_profiles.loc[gts_ts, 'dac_timestamp'] = dac_profiles.iloc[dac_timestamps.index(gts_ts)].name
        aligned_profiles.loc[gts_ts, 'dac_latitude'] = dac_profiles.iloc[dac_timestamps.index(gts_ts)].latitude
        aligned_profiles.loc[gts_ts, 'dac_longitude'] = dac_profiles.iloc[dac_timestamps.index(gts_ts)].longitude
        aligned_profiles.loc[gts_ts, 'dac_profile_id'] = dac_profiles.iloc[dac_timestamps.index(gts_ts)].profile_id

    aligned_profiles.index.name = 'gts_timestamp'

    # Rename the GTS lat/lon
    aligned_profiles = aligned_profiles.rename(columns={'latitude': 'gts_latitude', 'longitude': 'gts_longitude'})

    return aligned_profiles
