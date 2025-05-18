#!/usr/bin/env python
import streamlit as st
from st_files_connection import FilesConnection
import boto3

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.write("# Path Processing! 👋")

AWS_REGION_NAME = st.secrets["AWS_DEFAULT_REGION"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = "amads"
REMOTE_DIR = "data/map/"
FILE_KEYS = ["raw.bag", "map_params.yaml"]

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

try:
    for file_key in FILE_KEYS:
        s3_client.download_file(S3_BUCKET_NAME, REMOTE_DIR+file_key, file_key)
    st.markdown(
    """
    Streamlit for data processing.

    **👈 Select a function from the sidebar**
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    """
    )
    st.write('Raw data downloaded')
    st.sidebar.success("Select a processing step.")
except Exception as e:
     st.error(f"Error: {e}")
