#!/usr/bin/env python
import streamlit as st
import numpy as np
import pandas as pd
import yaml, pyproj
from lib_path_functions import downsample_path

# bagfile = 'raw.bag'
# gps_topic = '/novatel/oem7/inspva'
latlon_file = 'latlon.txt'
map_config_file = 'map_params.yaml'
latlon_file_processed = 'latlon_processed.txt'

downsample_spacing = st.number_input("Spacing", value=2.0, min_value=2.0)
trim_end = st.number_input("Trim end of path", value=0, min_value=0)

try:
    # timestamps, latitude, longitude, speed = read_inspva(bagfile, gps_topic)
    # np.savetxt(latlon_file, np.column_stack((timestamps, latitude, longitude, speed)), fmt='%.8f', delimiter=',')
    # st.write(f'Global coords file saved at: {latlon_file}')

    with open(map_config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    REF_LAT, REF_LON = config['projection']['reflat'], config['projection']['reflon']
    PROJ_K, PROJ_X, PROJ_Y = config['projection']['k'], config['projection']['x'], config['projection']['y']
    PROJ_STRING = f"+proj=tmerc +ellps=WGS84 +lat_0={REF_LAT} +lon_0={REF_LON} +k={PROJ_K} +x_0={PROJ_X} +y_0={PROJ_Y}  +units=m +no_defs"
    PROJ_PROJ = pyproj.Proj(PROJ_STRING)

    # read lat lon file
    latlon_data = np.loadtxt(latlon_file, delimiter=',')
    timestamps, lat, lon, speed = latlon_data[:,0], latlon_data[:,1], latlon_data[:,2], latlon_data[:,3]
    east, north = PROJ_PROJ(lon, lat)

    # downsample and resample
    east_ds, north_ds, coords_ds, ind_ds = downsample_path(east, north, spacing=downsample_spacing)
    lat_ds, lon_ds, speed_ds, timestamps_ds = lat[ind_ds], lon[ind_ds], speed[ind_ds], timestamps[ind_ds]

    # trim
    i1 = 0
    i2 = len(ind_ds) - trim_end
    east_ds, north_ds, lat_ds, lon_ds, speed_ds, timestamps_ds = \
        east_ds[i1:i2], north_ds[i1:i2], lat_ds[i1:i2], lon_ds[i1:i2], speed_ds[i1:i2], timestamps_ds[i1:i2]
    
    # write processed global file
    pt_size = np.array(lat_ds)*0.0+1.0
    pt_size[0] = 2.0
    latlon_df = pd.DataFrame(np.column_stack((lat_ds, lon_ds, pt_size)), columns=["lat", "lon", "size"])
    np.savetxt(latlon_file_processed, np.column_stack((timestamps_ds, lat_ds, lon_ds, east_ds, north_ds, speed_ds)), fmt='%.7f', delimiter=',')
    st.write(f'Processed global coords saved at: {latlon_file_processed}')

    # map
    st.map(latlon_df, latitude="lat", longitude="lon", size="size")
    
except Exception as e:
     st.error(f"Error: {e}")

