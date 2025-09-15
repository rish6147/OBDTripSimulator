import streamlit as st
import pandas as pd
import numpy as np
import requests # For OSRM API calls
from datetime import datetime, timedelta
import random
import folium # For map visualization
from streamlit_folium import st_folium # To display folium maps in Streamlit
import io  # For Excel download handling