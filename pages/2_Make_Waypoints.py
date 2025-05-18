#!/usr/bin/env python
import streamlit as st
import numpy as np
import pandas as pd
import yaml, pyproj
# from threading import RLock
from lib_path_functions import resample_route, smooth_waypoints, write_waypoint_file
# from bokeh.plotting import figure
import plotly.express as px

latlon_file = 'latlon_processed.txt'
map_config_file = 'map_params.yaml'
route_file = 'route.txt'
wp_file = 'waypoints.txt'

with st.sidebar:
    path_spacing = st.number_input("Spacing", value=1.0, min_value=0.0)
    zero_speed = st.number_input("Zero speed", value=1.5, min_value=0.0)
    path_end_tol = st.number_input("Path end tol.", value=0.1, min_value=0.1)
    smoothing_frame = st.number_input("smoothing frame", value=1.0, min_value=1.0)
    smoothing_variance = st.number_input("smoothing factor", value=0.1, min_value=0.01)

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
    timestamps, lat, lon, east, north, speed = latlon_data[:,0], latlon_data[:,1], latlon_data[:,2], latlon_data[:,3], latlon_data[:,4], latlon_data[:,5]

    # resample and smooth
    east_rs, north_rs, speed_rs, dist_rs, hdg_rs = resample_route(east, north, speed, path_spacing=path_spacing, path_end_tol=path_end_tol, zero_speed=zero_speed)

    # smooth path
    d_raw, d_fit, x_fit, y_fit, h_fit, v_fit = smooth_waypoints(east_rs, north_rs, speed_rs, dx_frame=smoothing_frame, path_variance=smoothing_variance)
    st.write(f"Average spacing after resampling and smoothing: {'%.2f' % np.mean(np.diff(d_fit))}")
    st.write(f"Total distance: {'%.2f' % d_fit[-1]} {'%.2f' % d_raw[-1]}")

    # write route file
    lon_fit, lat_fit = PROJ_PROJ(x_fit, y_fit, inverse=True)
    pt_size = np.array(lat_fit)*0.0+1.0
    pt_size[0] = 2.0
    latlon_df = pd.DataFrame(np.column_stack((lat_fit, lon_fit, pt_size)), columns=["lat", "lon", "size"])
    np.savetxt(route_file, np.column_stack((x_fit, y_fit)), fmt='%.3f', delimiter=',')
    st.write(f'Route file saved at: {route_file}')

    # write waypoint file
    write_waypoint_file(wp_file, x_fit, y_fit, v_fit, units='mps')
    st.write(f'Waypoints file saved at: {wp_file}')

    # # speed dataframe
    # speed_df = pd.DataFrame(np.column_stack((dist_rs, speed_rs)), columns=["dist", "speed"])
    speed_df = pd.DataFrame({'dist': d_fit, 'speed': v_fit})

    # map
    st.map(latlon_df, latitude="lat", longitude="lon", size="size", use_container_width=False, width=800, height=800)

    # bokey plot
    # p = figure(title="Speed vs. Distance", x_axis_label="Distance (m)", y_axis_label="Speed (m/s)")
    # p.line(d_fit, v_fit, legend_label="", line_width=2, line_color='blue')
    # st.bokeh_chart(p)

    # plotly plot
    fig = px.scatter(speed_df, x='dist', y='speed', title='Speed vs. Distance')
    fig.update_layout(
        xaxis_title="Distance (m)",
        yaxis_title="Speed (m/s)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
    )
    st.plotly_chart(fig, use_container_width=False, width=800, height=500)
    
except Exception as e:
     st.error(f"Error: {e}")

