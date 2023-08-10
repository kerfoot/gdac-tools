"""
Scratch space and notes for cartopy to build map and plot glider tracks
"""

import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set up logger
log_level = getattr(logging, 'INFO')
log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(format=log_format, level=log_level)

central_longitude = None
projection = 'PlateCarree'
global_map = False
bbox = None

bbox = [client.datasets.lon_min.min(),
        client.datasets.lon_max.max(),
        client.datasets.lat_min.min(),
        client.datasets.lat_max.max()]

if not central_longitude:
    central_longitude = client.datasets[['lon_min', 'lon_max']].mean().mean()

kws = {'projection': getattr(ccrs, projection)(central_longitude=central_longitude)}
map_fig, map_ax = plt.subplots(figsize=(11,8), subplot_kw=kws)

if not global_map:
    logging.info('Setting extent to bbox')
    map_ax.set_extent(bbox)
else:
    logging.info('Setting extent to global')
    map_ax.set_global()

edge_color = "black"
land_color = "tan"
ocean_color = cfeature.COLORS['water']  # cfeature.COLORS['water'] is the standard
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

marker = 'None'
marker_size = 1.0
linestyle = '-'
cbar = mpl.cm.get_cmap('rainbow', client.datasets.shape[0])
i = 0
for dataset_id, row in client.datasets.iterrows():

    gps = client.daily_profile_positions[client.daily_profile_positions.dataset_id == dataset_id]
    if gps.empty:
        logging.warning('No GPS track found for {:}'.format(dataset_id))

    track = gps.sort_values('date', ascending=True)

    # Plot the track
    map_ax.plot(track.longitude, track.latitude, marker=marker, markersize=marker_size, linestyle=linestyle, color=cbar(i),
                  transform=ccrs.PlateCarree())

    i += 1
