#!/usr/bin/env python
import streamlit as st
import numpy as np
from st_files_connection import FilesConnection
import boto3, rosbag, yaml, pyproj
from io import BytesIO

def read_inspva(bagfile, gps_topic):
	import rosbag
	timestamps, latitude, longitude, speed = [], [], [], []
	bag = rosbag.Bag(bagfile)
	for topic, msg, t in bag.read_messages(topics=[gps_topic]):
		#t_topic = msg.header.stamp.secs + 1E-9*msg.header.stamp.nsecs - t0
		t_topic = t.to_sec()
		if topic == gps_topic:
			latitude.append(msg.latitude)
			longitude.append(msg.longitude)
			north_velocity = msg.north_velocity
			east_velocity = msg.east_velocity
			speed.append(np.hypot(north_velocity, east_velocity))
			timestamps.append(t_topic)
	return timestamps, latitude, longitude, speed

def downsample_path(x, y, spacing=1.0):
	x_downsampled, y_downsampled = [], []
	coords_downsampled = []
	indices_downsampled = []

	for i in range(0, len(x)):                
		if i == 0: # first point so just accept whatever gps position and init pp
			x_downsampled.append(x[i])
			y_downsampled.append(y[i])
			x_prev, y_prev = x[i], y[i]
			coords_downsampled.append((x[i], y[i]))
			indices_downsampled.append(i)

		elif i == len(x)-1: # if last point, just append
			x_downsampled.append(x[i])
			y_downsampled.append(y[i])
			coords_downsampled.append((x[i], y[i]))
			indices_downsampled.append(i)
		
		else: # if not first or last then check delta path
			dx = np.hypot(x[i]-x_prev, y[i]-y_prev)
			if dx >= spacing: # append if path spacing exceeds desired spacing
				x_downsampled.append(x[i])
				y_downsampled.append(y[i])
				coords_downsampled.append((x[i], y[i]))
				indices_downsampled.append(i)
				x_prev, y_prev = x[i], y[i]

	return x_downsampled, y_downsampled, coords_downsampled, indices_downsampled

bagfile = 'raw.bag'
gps_topic = '/novatel/oem7/inspva'
latlon_file = 'latlon.txt'
map_config_file = 'map_params.yaml'
downsample_spacing = 1.0
route_file = 'route.txt'

try:
    timestamps, latitude, longitude, speed = read_inspva(bagfile, gps_topic)
    np.savetxt(latlon_file, np.column_stack((timestamps, latitude, longitude, speed)), fmt='%.8f', delimiter=',')
    st.write(f'Global coords file saved at: {latlon_file}')

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
	
    # write route file
    np.savetxt(route_file, np.column_stack((east_ds, north_ds)), fmt='%.3f', delimiter=',')
    st.write(f'Route file saved at: {route_file}')
	
except Exception as e:
     st.error(f"Error: {e}")

