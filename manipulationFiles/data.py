#! /usr/bin/env python
"""
    This script calculates distances between various PSA terminals and the locality that we pinpoint.
"""

from ipywidgets import widgets, Layout, Button, Output, VBox, DatePicker, HTML
from IPython.display import display, clear_output
import pandas as pd
import folium
import googlemaps
from datetime import datetime
import random
import math
from math import radians, cos, sin, asin, sqrt
import numpy as np
from gurobipy import Model, GRB, quicksum
from folium import IFrame
import webbrowser
from io import StringIO
import json
import requests
import time
from folium.plugins import MousePosition
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
GOOGLE_API_KEY = 'REENACT'
searoutes_API_KEY = 'REENACT'


# PSA Terminals arnd the world coordinates (Asia & Europe)
PSAPORTDICT={
    'Singapore': {
        'Coord':[1.3521,103.8198],
        'Tuas Port': [1.2393875726212689, 103.61601811280622],
        'Brani Terminal': [1.262453303337454, 103.83277461719618],
        'Pasir Panjang Terminal': [1.2876741671123324, 103.77384253237409],
        'Jurong Island Terminal': [1.2774376052999765, 103.67008186791996],
    },
    'Vietnam': {
        'Coord':[14.0583, 108.2772],
        'SP-PSA International Terminal': [10.566988637391177, 107.02392880211224],
        'Tan Cang Que Vo Inland Container Depot': [10.8833,106.7333],
    },
    'Thailand': {
        'Coord':[15.8700,100.9925],
        'Laem Chabang Port': [13.0660799,100.8643659],
        'Bangkok Port': [13.6517128,100.5529994],
    },
    'Indonesia': {
        'Coord':[-0.7893,113.9213],
        'NEW PRIOK CONTAINER TERMINAL ONE': [-6.0946545315733385, 106.92315568836017],
    },
    'Belgium': {
        'Coord':[50.5039,4.4699],
        'Port of Antwerp': [51.23036731274294, 4.4113027537855976],
        'PSA ZEEBRUGGE': [51.34984239048196, 3.1740156691349526]
    },
    'Italy': {
        'Coord':[41.8719,12.5674],
        'PSA Venice (Vecon)': [45.45869518573302, 12.245117413491945],
        'PSA Genova PRA': [44.424147218241515, 8.781699836185101],
        # 'PSA SECH': [44.417407658056824, 8.911251940109276],
    },
    'Portugal': {
        'Coord':[39.3999,-8.2245],
        'PSA Sines': [37.93867706093018, -8.8444355],
    },
    'Turkey': {
        'Coord':[38.9637,35.2433],
        'PSA Akdeniz': [36.806115328282615, 34.639095499252896],
    },
    'Poland': {
        'Coord':[51.9194,19.1451],
        'PSA Gdansk': [54.38163749997303, 18.7128604213871],
    },
    'China':{
        'Coord':[34.7494057238321, 102.95195840172003],
        'Dalian Container Terminal':[39.00283412804412, 121.88036160895096],
        'Fuzhou International Container Terminal':[25.44001884731009, 119.37498488002498],
        'Guangzhou South China Oceangate Container Terminal':[22.637147632440925, 113.67682939750225],
        'Tianjin Port Container Terminal':[39.12577211246826, 117.79930007666104],
        'Lianyungang Gangguanhe Port':[34.744225601821114, 119.39080274158614],
        'PSA Qinzhou': [21.72728717807111, 108.58876247849621]
    },
    'Korea':{
        'Coord':[36.61277405429985, 128.25818685857573],
        'Hanjin Incheon Container Terminal':[39.00283412804412, 121.88036160895096],
        'Busan Port Terminal':[35.11799252571318, 129.10348456132897],
    },
    'Japan':{
        'Coord':[36.511407153934556, 138.01681946897156],
        'Hibiki Container Terminal':[33.89796852900625, 130.9205659052803],
    },
}

# Function to get the address from coordinates
def get_address(loca):
    location = geolocator.reverse((loca), exactly_one=True, language='en')
    if location:
        address = location.raw['address']
        return address
    else:
        return None

# Function to get coordinates from a location name
def get_coordinates(location_name):
    location = geolocator.geocode(location_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

# Function to calculate distance between two locations (in km)
def calculate_distance(start_location, end_location):
    start_coords = get_coordinates(start_location)
    end_coords = get_coordinates(end_location)
    if start_coords and end_coords:
        return geodesic(start_coords, end_coords).kilometers
    else:
        return None

# Load the data (duisburg to chongqing estimated train distance is 11,179km)
links_df = pd.read_csv('mincost.csv')
b_df = pd.read_csv('mincost_b.csv')

# Initialize the Nominatim geocoder
geolocator = Nominatim(user_agent="test", timeout=10)

# # Calculate distances and update the DataFrame
# links_df['c_ij'] = links_df.apply(lambda row: calculate_distance(row['start node i'], row['end node j']), axis=1)

# Calculate distances and update the DataFrame only if c_ij is 0
links_df['c_ij'] = links_df.apply(
    lambda row: calculate_distance(row['start node i'], row['end node j']) if row['c_ij'] == 0 else row['c_ij'],
    axis=1
)
output_file = 'links_df.csv'
links_df.to_csv(output_file, index=False)

b_df['latitude'], b_df['longitude'] = zip(*b_df['node i'].apply(get_coordinates))

b_df.loc[b_df['node i'] == 'Qinzhou', ['latitude','longitude']] = PSAPORTDICT['China']['PSA Qinzhou']

# Create the coordinates column
b_df['coordinates'] = b_df.apply(lambda row: [row['latitude'], row['longitude']], axis=1)

output_file2 = 'b_df.csv'
b_df.to_csv(output_file2, index=False)