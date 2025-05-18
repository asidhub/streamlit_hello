#!/usr/bin/env python
import streamlit as st
import numpy as np
from st_files_connection import FilesConnection
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

latlon_file = 'latlon.txt'
route_file = 'route.txt'

try:
    s3_client.upload_file(latlon_file, S3_BUCKET_NAME, REMOTE_DIR+latlon_file)
    s3_client.upload_file(latlon_file, S3_BUCKET_NAME, REMOTE_DIR+route_file)
    st.write('Processed data uploaded')
except Exception as e:
     st.error(f"Error: {e}")

