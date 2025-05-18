#!/usr/bin/env python
import streamlit as st
import numpy as np
import boto3

AWS_REGION_NAME = st.secrets["AWS_DEFAULT_REGION"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = "amads"
REMOTE_DIR = 'data/map/'

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# latlon_file = 'latlon.txt'
route_file = 'route.txt'
wp_file = 'waypoints.txt'

try:
    s3_client.upload_file(route_file, S3_BUCKET_NAME, REMOTE_DIR+route_file)
    s3_client.upload_file(wp_file, S3_BUCKET_NAME, REMOTE_DIR+wp_file)
    st.write(f'Processed files uploaded: {route_file}, {wp_file}')
except Exception as e:
    st.error(f"Error: {e}")

