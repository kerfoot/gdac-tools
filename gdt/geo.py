import requests
import logging
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from decimal import *

dac_catalog_url = 'https://gliders.ioos.us/providers/api/deployment'

logging.getLogger(__file__)


def df2geodf(datasets, crs='EPSG:4326'):
    """
    Create a GeoDataFrame from the datasets DataFrame using lat_min, lat_max, lon_min, lon_max coordinates

    :param datasets: DAC datasets DataFrame
    :param crs: coordinate reference system
    :return: GeoDataFrame
    """

    gdf = GeoDataFrame()

    coords = ['lat_min',
              'lat_max',
              'lon_min',
              'lon_max']
    has_coords = True
    for coord in coords:
        if coord not in datasets:
            logging.error('Missing coordinate {:}'.format(coord))
            has_coords = False
            continue

        datasets[coord] = datasets[coord].astype('float')

    if not has_coords:
        return gdf

    # polygon = [nw, ne, se, sw, nw]
    bboxes = []
    for _, row in datasets.iterrows():
        bbox = Polygon()
        if not pd.isna(row.lat_min):
            bbox = Polygon(((row.lat_max, row.lon_max),
                            (row.lat_max, row.lon_min),
                            (row.lat_min, row.lon_min),
                            (row.lat_min, row.lon_max),
                            (row.lat_max, row.lon_max)))
        bboxes.append(bbox)

    gdf = GeoDataFrame(datasets, geometry=bboxes, crs=crs)

    return gdf


def latlon_to_geojson_track(latitudes, longitudes, timestamps, include_points=True, precision='0.001'):

    geojson = {'type': 'FeatureCollection',
               'bbox': latlon_to_bbox(latitudes, longitudes, timestamps, precision=precision)}

    features = [latlon_to_linestring(latitudes, longitudes, timestamps, precision=precision)]

    if include_points:
        points = latlon_to_points(latitudes, longitudes, timestamps, precision=precision)
        features = features + points

    geojson['features'] = features

    return geojson


def latlon_to_linestring(latitudes, longitudes, timestamps, precision='0.001'):
    dataset_gps = pd.DataFrame(index=timestamps)
    dataset_gps['latitude'] = latitudes.values
    dataset_gps['longitude'] = longitudes.values

    track = {'type': 'Feature',
             # 'bbox': bbox,
             'geometry': {'type': 'LineString',
                          'coordinates': [
                              [float(Decimal(pos.longitude).quantize(Decimal(precision),
                                                                     rounding=ROUND_HALF_DOWN)),
                               float(Decimal(pos.latitude).quantize(Decimal(precision),
                                                                    rounding=ROUND_HALF_DOWN))]
                              for i, pos in dataset_gps.iterrows()]},
             'properties': {}
             }

    return track


def latlon_to_points(latitudes, longitudes, timestamps, precision='0.001'):
    dataset_gps = pd.DataFrame(index=timestamps)
    dataset_gps['latitude'] = latitudes.values
    dataset_gps['longitude'] = longitudes.values

    return [{'type': 'Feature',
             'geometry': {'type': 'Point', 'coordinates': [float(Decimal(pos.longitude).quantize(Decimal(precision),
                                                                                                 rounding=ROUND_HALF_DOWN)),
                                                           float(Decimal(pos.latitude).quantize(Decimal(precision),
                                                                                                rounding=ROUND_HALF_DOWN))]},
             'properties': {'ts': i.strftime('%Y-%m-%dT%H:%M:%SZ')}}
            for i, pos in dataset_gps.iterrows()]


def latlon_to_bbox(latitudes, longitudes, timestamps, precision='0.001'):
    dataset_gps = pd.DataFrame(index=timestamps)
    dataset_gps['latitude'] = latitudes.values
    dataset_gps['longitude'] = longitudes.values

    return [float(Decimal(dataset_gps.longitude.min()).quantize(Decimal(precision), rounding=ROUND_HALF_DOWN)),
            float(Decimal(dataset_gps.latitude.min()).quantize(Decimal(precision), rounding=ROUND_HALF_DOWN)),
            float(Decimal(dataset_gps.longitude.max()).quantize(Decimal(precision), rounding=ROUND_HALF_DOWN)),
            float(Decimal(dataset_gps.latitude.max()).quantize(Decimal(precision), rounding=ROUND_HALF_DOWN))]
