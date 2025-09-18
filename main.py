import streamlit as st
import pandas as pd
import numpy as np
import requests # For OSRM API calls
from datetime import datetime, timedelta
import random
import folium # For map visualization
from streamlit_folium import st_folium # To display folium maps in Streamlit
import io  # For Excel download handling

#---Bounding Box Taiwan main Island --- 
TAIWAN_LAT_MIN = 21.8
TAIWAN_LAT_MAX = 25.3
TAIWAN_LON_MIN = 120.0
TAIWAN_LON_MAX = 121.8 
  
#OSRM public API endpoint
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/"

#----Creating the dictonary for the Major Cities in taiwan with their coordinates----
TAIWAN_CITIES = {
    "Taipei": (25.0330, 121.5654),
    "Kaohsiung": (22.6273, 120.3014),
    "Taichung": (24.1477, 120.6736),
    "Tainan": (22.9999, 120.2270),
    "Hsinchu": (24.8138, 120.9675),
    "Keelung": (25.1276, 121.7392),
    "Chiayi": (23.4800, 120.4500),
    "Yilan": (24.7587, 121.7570),
    "Hualien": (23.9871, 121.6015),
    "Taitung": (22.7583, 121.1446)
}
#------------------------------------------------------------------------------------