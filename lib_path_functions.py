#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate
import csv

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

def nearest_lower_index(arr, target):
	nearest_index = -1
	nearest_value = float('-inf')  # Initialize to negative infinity
	
	for i in range(len(arr)):
		if arr[i] <= target:
			if arr[i] > nearest_value:
				nearest_value = arr[i]
				nearest_index = i

	return nearest_index  

def get_min(input_list):
	# min_value, min_index = get_min(np.abs(np.array(s_time)-item['event_start_time']))
	min_value = np.amin(np.array(input_list))
	return min_value, np.where(input_list == min_value)[0][0]

class pathly():
	def __init__(self, x, y, proj_proj=None):
		# self.xy = np.loadtxt(route_file,delimiter=',')
		if proj_proj == None:
			self.xy = np.array([(x[i], y[i]) for i in range(0, len(x))])
		else:
			self.xy = np.array([proj_proj(y[i], x[i]) for i in range(0, len(x))])
		self.npoints = len(self.xy)
		self.compute_dist()
		self.route_len = max(self.d)
		self.dist_now = None
		self.id_now = None
		self.initialized = False
		# all track heading
		self.track_hdg = []
		for i in range(0, len(self.xy)-1):
			self.track_hdg.append(np.arctan2(self.xy[i+1][1]-self.xy[i][1], self.xy[i+1][0]-self.xy[i][0]))
		self.track_hdg.append(self.track_hdg[-1])
		
	def proj_on_seg(self,segment_start,segment_end,point):
		segment_vector = segment_end - segment_start

		# If the segment is just a point, return either start or end
		if np.all(segment_vector == 0):
			projection = segment_start
			t=0
		else:
			point_vector = point - segment_start
			
			# Calculate the projection parameter t
			t = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
			
			# If t is not within the segment (0 <= t <= 1), clamp it
			t = max(0, min(1, t))
			
			# Calculate the projection
			projection = segment_start + t * segment_vector
		
		dist = np.linalg.norm(projection-point,ord=2)
		
		return projection, dist,t
	
	def compute_dist(self):
		d = [0]
		dist = 0
		delta_d = []
		for i in range(self.npoints-1):    # i=0 has zero distnace, the first distance computed is for the second point (i+1)
			delta = np.linalg.norm(self.xy[i+1]-self.xy[i],ord=2)
			dist = dist+delta
			d.append(dist)
			delta_d.append(delta)
		delta_d.append(np.linalg.norm(self.xy[self.npoints-1]-self.xy[0],ord=2))  # last point to first point
		self.d = d
		self.delta_d = delta_d


	def get_dist(self,target):
		dist_max = 100  # a very large value (>1 for most cases)
		dist =np.nan
		if self.dist_now is None:
			for i in range(self.npoints-1):
				proj,perp_dist,t = self.proj_on_seg(self.xy[i],self.xy[i+1],target)
				if perp_dist<dist_max:
					dist_max=perp_dist
					dist = self.delta_d[i]*t+self.d[i]
					self.id_now = i
			
		else:
			for k in range(10):
				i=k+self.id_now-1
				i = i%self.npoints
				if i == self.npoints-1:
					proj,perp_dist,t = self.proj_on_seg(self.xy[i],self.xy[0],target)    
				else:
					proj,perp_dist,t = self.proj_on_seg(self.xy[i],self.xy[i+1],target)
				if perp_dist<dist_max:
					dist_max=perp_dist
					dist = self.delta_d[i]*t+self.d[i]
					self.id_now = i
		self.dist_now = dist
		self.initialized = True
		return dist

	def get_xy(self,target_d):
		target_d = target_d%self.d[-1]
		id = nearest_lower_index(self.d,target_d)
		
		xy = self.xy[id]
		if id==self.npoints-1:
			xy_next = self.xy[0]
		else:
			xy_next = self.xy[id+1] 
		d0 = self.d[id]
		delta_d = self.delta_d[id]
		t = (target_d-d0)/delta_d
		h0 = self.track_hdg[id]
		h1 = self.track_hdg[(id+1) % self.npoints]
		delta_h = h1-h0
		if delta_h > np.pi:
			delta_h = delta_h - 2*np.pi
		if delta_h < -np.pi:
			delta_h = delta_h + 2*np.pi
		heading = h0 + delta_h*t
		if heading > np.pi:
			heading = heading - 2*np.pi
		if heading < -np.pi:
			heading = heading + 2*np.pi
		return (xy_next-xy)*t+xy, heading*180/np.pi
	
	
	def reset(self):
		self.dist_now = None
		self.id_now = None
		self.initialized = False


def resample_route(east_ds, north_ds, speed_ds, path_spacing=1.0, path_end_tol=0.1, zero_speed=1.5):
	path_sampler = pathly(x=east_ds, y=north_ds) # proj_proj=PROJ_PROJ
	dist_ds = path_sampler.d
	hdg_ds = path_sampler.track_hdg
	d_array = np.arange(0.0, path_sampler.route_len//path_spacing+path_spacing, path_spacing)
	d_array = np.append(d_array, path_sampler.route_len-path_end_tol)
	# print(f'End of path tolerance: {path_end_tol}')
	x_arr, y_arr, h_arr = [], [], []
	for d in d_array:
		xy_track, h = path_sampler.get_xy(d)
		x, y = xy_track[0], xy_track[1]
		x_arr.append(x)
		y_arr.append(y)
		h_arr.append(h)
	dist_rs = d_array
	east_rs = x_arr
	north_rs = y_arr
	hdg_rs = h_arr

	# resample
	speed_rs = []
	for d in dist_rs:
		speed_rs.append(np.interp(d, dist_ds, speed_ds))
	if abs(speed_rs[-1]) < zero_speed:
		print(f'Speed at end of route within zero margin, setting to 0.0')
		speed_rs[-1] = 0.0

	return east_rs, north_rs, speed_rs, dist_rs, hdg_rs

def spline_curvature(tck, unew):
    dx, dy = interpolate.splev(unew, tck, der=1)
    ddx, ddy = interpolate.splev(unew, tck, der=2)
    K = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))
    hdg = np.arctan2(dy, dx)
    return K, hdg

def spline_fit(x_raw, y_raw, ds, smoothing=None):
    if smoothing is not None:
        tck, u = interpolate.splprep([x_raw, y_raw], s=smoothing)
    else:
        tck, u = interpolate.splprep([x_raw, y_raw])
    u_fit = np.arange(0, 1+ds, ds)
    out = interpolate.splev(u_fit, tck)
    x_fit, y_fit = out[0], out[1]
    k_fit, h_fit = spline_curvature(tck, u_fit)
    return x_fit, y_fit, k_fit, h_fit

def smooth_waypoints(x_raw, y_raw, v_raw, dx_frame=1, path_variance=0.1):
    Nt = len(x_raw)
    ds = 1/(Nt*dx_frame) # n points in spline fit are approx 1 ft apart
    x_fit, y_fit, k_fit, h_fit = spline_fit(x_raw, y_raw, ds, smoothing=(path_variance**2)*len(x_raw))

    # Path sampler based on smooth data
    path_sampler = pathly(x=x_fit, y=y_fit) # proj_proj=PROJ_PROJ
    d_fit = path_sampler.d

    # Get projected distances of raw data on path sampler
    d_raw = []
    for i, (x,y) in enumerate(zip(x_raw, y_raw)):
        d_raw.append(path_sampler.get_dist((x, y)))

    # Get speed at each of the smooth dists by interpolating raw speeds vs raw dist
    v_fit = []
    for d in d_fit:
        v_fit.append(np.interp(d, d_raw, v_raw))

    return d_raw, d_fit, x_fit, y_fit, h_fit, v_fit


def write_waypoint_file(filepath, x, y, v, units='kph'):
	with open(filepath, "w", encoding='UTF8', newline='') as file:
		headers = ['x', 'y', 'z', 'yaw', 'velocity', 'change_flag']
		writer = csv.writer(file)
		writer.writerow(headers)
		for i in range(0, len(x)):
			if i != len(x)-1:
				hdg = np.arctan2(y[i+1]-y[i], x[i+1]-x[i])
			if units == 'mps':
				speed = v[i] * 18/5
			else:
				speed = v[i]
			row = [x[i], y[i], 0.0, hdg, speed, 0]
			row = ['%.2f' % row[0], '%.2f' % row[1], '%.2f' % row[2], '%.5f' % row[3], '%.2f' % row[4], '%d' % row[5]]
			writer.writerow(row)
	file.close()
	# print(f"Waypoints file created at: {filepath}")