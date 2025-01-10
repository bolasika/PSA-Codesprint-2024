#! /usr/bin/env python
'''
    This script includes the algorithms taht calculates optimized routes for delivery using various transportation methods.
    It includes functions for calculating distances, querying APIs, and plotting routes on a map.
'''
# TODO: Enter the API Keys
GOOGLE_API_KEY = 'REDACTED'
searoutes_API_KEY = 'REDACTED'


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
geolocator = Nominatim(user_agent="test", timeout=10)

# PSA Terminals arnd the world coordinates (asia & europe)
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
PSAPORTDICT['China']['PSA Qinzhou']=[21.72728717807111, 108.58876247849621]

# Function to calculate the Haversine distance for airplane + ship
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def closestPort(psaDict, target):
    min_coord=[]
    min_distance = 'a'
    for eachKey in psaDict:
        if min_distance == 'a':
            min_coord = psaDict[eachKey]
            min_distance = haversine(target[0],target[1], psaDict[eachKey]['Coord'][0], psaDict[eachKey]['Coord'][1])
        else:
            distance = haversine(target[0],target[1], psaDict[eachKey]['Coord'][0], psaDict[eachKey]['Coord'][1])
            min_distance = min(min_distance, distance)
            if min_distance == distance:
                min_coord = psaDict[eachKey]
    min_distance = 'a'
    target_port = []
    for eachKey in min_coord:
        if eachKey != 'Coord':
            if min_distance == 'a':
                target_port = min_coord[eachKey]
                min_distance = haversine(target[0],target[1], min_coord[eachKey][0], min_coord[eachKey][1])
            else:
                distance = haversine(target[0],target[1], min_coord[eachKey][0], min_coord[eachKey][1])
                min_distance = min(min_distance, distance)
                if min_distance == distance:
                    target_port = min_coord[eachKey]
    return target_port

def get_route_distance(data, data_panama=None, data_suez=None):
    route_distance=data['features'][0]['properties']['distance'] / 1000
    route_duration=data['features'][0]['properties']['duration']/8.64e+7
    if data_panama != None:
        panama_flag=True
        panama_route_distance=data_panama['features'][0]['properties']['distance'] / 1000
        panama_route_duration=data_panama['features'][0]['properties']['duration']/8.64e+7
    else:
        panama_flag=False
    if data_suez != None:
        suez_flag=True
        suez_route_distance=data_suez['features'][0]['properties']['distance'] / 1000
        suez_route_duration=data_suez['features'][0]['properties']['duration']/8.64e+7
    else:    
        suez_flag=False

    if panama_flag==True and suez_flag==True:
        return [[route_distance, route_duration], [panama_route_distance, panama_route_duration], [suez_route_distance, suez_route_duration]]
    elif panama_flag==True:
        return [[route_distance, route_duration], [panama_route_distance, panama_route_duration],[]]
    elif suez_flag==True:
        return [[route_distance, route_duration], [], [suez_route_distance, suez_route_duration]]
    else:
        return [[route_distance, route_duration],[],[]]

def searoutes_query(coord, searoutes_API_KEY=searoutes_API_KEY, avoidHRA=False, avoidSeca=False, allowIceAreas=False, blockAreas=[]):
    # Initialize data as None
    data = None

    # API call for searoutes to get the route
    url = f'https://api.searoutes.com/route/v2/sea/{coord[0][1]}%2C{coord[0][0]}%3B{coord[1][1]}%2C{coord[1][0]}'
    params = {
        'continuousCoordinates': 'true',
        'allowIceAreas': allowIceAreas,
        'avoidHRA': avoidHRA,
        'avoidSeca': avoidSeca,
    }
    if blockAreas:
        params['blockAreas'] = blockAreas

    headers = {
        'accept': 'application/json',
        'x-api-key': searoutes_API_KEY
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        # Successful response
        data = response.json()
        print("Searoutes data retrieved successfully.")
    elif response.status_code == 429:
        print("Error: Rate limit exceeded for Searoutes API.")
    else:
        # Other error response
        print(f"Error {response.status_code}: {response.text}")

    return data


def area_data(data):
    # Extract the LineString coordinates and the point for areas
    line_coords = data['features'][0]['geometry']['coordinates']
    start_coord = line_coords[0]
    end_coord = line_coords[-1]
    if len(data['features'][0]['properties']['areas']['features'])>1:
        area_coords =[]
        area_names = []
        for i,_ in enumerate(data['features'][0]['properties']['areas']['features']):
            area_coords.append(data['features'][0]['properties']['areas']['features'][i]['geometry']['coordinates'])
            area_names.append(data['features'][0]['properties']['areas']['features'][i]['properties']['name'])

    else:
        area_coords = data['features'][0]['properties']['areas']['features'][0]['geometry']['coordinates']
        area_names = data['features'][0]['properties']['areas']['features'][0]['properties']['name']
    print(area_names)
    print(area_coords)
    return line_coords, start_coord, end_coord, area_coords, area_names

def create_map(line_coords):
    # Set up the initial map location to the first point in the LineString
    return folium.Map(location=[line_coords[0][1], line_coords[0][0]], zoom_start=12)

def plot_map(line_coords, start_coord, end_coord, area_coords, area_names,num):
    colors=['blue', 'darkblue','orange']
    # Add the route (LineString) to the map
    folium.PolyLine(locations=[(lat, lon) for lon, lat in line_coords],
                    color=colors[num],
                    weight=3,
                    opacity=0.7).add_to(m)

    # Add a marker for the start and end points
    folium.Marker(location=[start_coord[1], start_coord[0]],
                popup='Starting Port',
                icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=[end_coord[1], end_coord[0]],
                popup='Destination Port',
                icon=folium.Icon(color='green')).add_to(m)

    # Add a marker for the regions
    if len(area_names)>1:
        for i,_ in enumerate(area_names):
            folium.Marker(location=[area_coords[i][1], area_coords[i][0]],
                popup=area_names[i],
                icon=folium.Icon(color='red')).add_to(m)
    else:
        folium.Marker(location=[area_coords[1], area_coords[0]],
                    popup=area_names,
                    icon=folium.Icon(color='red')).add_to(m)

def check_suez_panama(area_names,coord):
    print(area_names)
    if 'Panama Canal' in area_names:
        time.sleep(1)
        data_panama = searoutes_query(coord, blockAreas=[11112])
        line_coords_panama, start_coord_panama, end_coord_panama, area_coords_panama, area_names_panama = area_data(data_panama)
        plot_map(line_coords_panama, start_coord_panama, end_coord_panama, area_coords_panama, area_names_panama,1)
    else:
        data_panama = None

    if 'Suez Canal' in area_names: 
        time.sleep(1)
        data_suez = searoutes_query(coord, blockAreas=[11117])
        line_coords_suez, start_coord_suez, end_coord_suez, area_coords_suez, area_names_suez = area_data(data_suez)
        plot_map(line_coords_suez, start_coord_suez, end_coord_suez, area_coords_suez, area_names_suez,2)
    else:  
        data_suez = None
    return data_panama, data_suez



# Calculates the distance matrix using Google Maps for a given DataFrame of locations.
def get_googlemaps_distance_matrix(df):
    """
    Calculates the distance matrix using Google Maps for a given DataFrame of locations.
    
    Parameters:
    - df: DataFrame containing the columns 'LATITUDE' and 'LONGITUDE'.
    
    Returns:
    - A NumPy array representing the distance matrix.
    """
    N = len(df)
    distance_matrix = np.zeros((N, N))
    
    # Initialize the client 
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                source = df.iloc[i]
                dest = df.iloc[j]
                
                # Make the request
                result = gmaps.distance_matrix((source['latitude'], source['longitude']),
                                               (dest['latitude'], dest['longitude']),
                                               mode="driving")
                
                # Extract the distance from the response
                distance = result['rows'][0]['elements'][0]['distance']['value']  # Distance in meters
                distance_matrix[i, j] = distance / 1000.0  # Convert to kilometers
    
    return distance_matrix

# TODO: add back m
def delivery_route(orders, depot_coords, depot_demand, K, C, m):    
    depot_data = {'latitude': depot_coords[0], 'longitude': depot_coords[1], 'volume': depot_demand}    # psa venice
    # Concatenate the depot data with the orders dataframe
    depot_df = pd.DataFrame([depot_data])
    orders = pd.concat([depot_df, orders], ignore_index=True)


    # Number of nodes (including depot)
    N = len(orders['latitude'])

    # Set of nodes, including depot (node 0)
    V = set(range(N))

    # Demand at each node (including depot with 0 demand)
    d = {i: orders['volume'][i] for i in V}
    

    # Distance matrix using Haversine formula
    # c = {(i, j): haversine(orders['latitude'][i], orders['longitude'][i], orders['latitude'][j], orders['longitude'][j])
    #      for i in V for j in V if i != j}
    
    # Distance matrix using Google Maps
    c = get_googlemaps_distance_matrix(orders)

    # Seed random number generator for reproducibility
    random.seed(1) 

    # Create a new model
    model = Model("Vehicle Routing Problem")

    # Decision variables
    x = model.addVars(K, V, V, vtype=GRB.BINARY, name='x')
    u = model.addVars(K, V, vtype=GRB.CONTINUOUS, name='u')

    # Objective function
    model.setObjective(sum(c[i, j] * x[k, i, j] for k in range(K) for i in V for j in V if i != j), GRB.MINIMIZE)

    # Constraints

    # Each node must be visited exactly once by exactly one vehicle
    for j in V - {0}:
        model.addConstr(sum(x[k, i, j] for k in range(K) for i in V if i != j) == 1)

    # Each node must be departed exactly once by exactly one vehicle
    for i in V - {0}:
        model.addConstr(sum(x[k, i, j] for k in range(K) for j in V if i != j) == 1, name=f"depart_{i}")

    # Number of vehicles leaving and entering the depot
    for k in range(K):
        model.addConstr(sum(x[k, 0, j] for j in V if j != 0) == 1)
        model.addConstr(sum(x[k, i, 0] for i in V if i != 0) == 1)

    # Flow conservation constraints
    for k in range(K):
        for j in V - {0}:
            model.addConstr(sum(x[k, i, j] for i in V if i != j) == sum(x[k, j, i] for i in V if i != j))

    # Vehicle load constraints
    for k in range(K):
        for i in V - {0}:
            model.addConstr(u[k, i] >= d[i], f'load_at_least_demand_{k}_{i}')
            model.addConstr(u[k, i] <= C, f'load_at_most_capacity_{k}_{i}')
            for j in V - {0}:
                if i != j:
                    model.addConstr(u[k, j] >= u[k, i] - C * (1 - x[k, i, j]) + d[j], f'load_increase_{k}_{i}_{j}')

    # Depot load
    for k in range(K):
        model.addConstr(u[k, 0] == 0)

    # No self-loops
    for k in range(K):
        for i in V:
            model.addConstr(x[k, i, i] == 0)

    model.setParam('OutputFlag', 0)

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Initialize Google Maps client
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
        now = datetime.now()  # Current time for routing purposes

        # Use a different color for each vehicle's route
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'lightred', 'beige', 'darkgreen', 'cadetblue']

        for k in range(K):
            current_location = depot_coords  # Start at the depot
            route_details = []

            for j in V:
                if j != 0 and x[k, 0, j].x > 0.5:  # Find the first node from the depot
                    next_node = j
                    break

            while next_node != 0:
                next_location = (orders.at[next_node, 'latitude'], orders.at[next_node, 'longitude'])
                # Fetch directions from the current location to the next location
                directions_result = gmaps.directions(current_location, next_location, mode="driving", departure_time=now)

                if directions_result:
                    polyline = directions_result[0]['overview_polyline']['points']
                    points = googlemaps.convert.decode_polyline(polyline)
                    folium.PolyLine([(point['lat'], point['lng']) for point in points], color=colors[k % len(colors)], weight=2.5, opacity=1).add_to(m)
                    current_location = next_location  # Update current location to the next location

                # Update to the next node
                i = next_node
                next_node = 0
                for j in V:
                    if x[k, i, j].x > 0.5:
                        next_node = j
                        break

            # Optionally, draw a line back to the depot
            directions_result = gmaps.directions(current_location, depot_coords, mode="driving", departure_time=now)
            if directions_result:
                polyline = directions_result[0]['overview_polyline']['points']
                points = googlemaps.convert.decode_polyline(polyline)
                folium.PolyLine([(point['lat'], point['lng']) for point in points], color=colors[k % len(colors)], weight=2.5, opacity=1, dash_array='5, 5').add_to(m)

        # Add a special marker for the depot
        html_warehouse = f"<div style='width:80px;'><strong>Warehouse</strong></div>"  # Adjust width as needed
        iframe_warehouse = IFrame(html_warehouse, width=80+10, height=50)  # Width includes a little extra for padding, height is as needed
        popup_warehouse = folium.Popup(iframe_warehouse, max_width=2650)  # max_width is set high to allow custom width to take effect
        folium.Marker(
            location=depot_coords,
            popup=popup_warehouse,
            icon=folium.Icon(color='red', icon='industry', prefix='fa')
        ).add_to(m)

        # Add markers for other coordinates
        for idx, row in orders.iterrows():
            if idx != 0:  # Exclude depot
                address = row['address']  # Retrieve the address from the DataFrame
                # HTML content with address included
                html = f"<div style='width:200px;'><strong>Order {idx}</strong><br/>{address}</div>"  # Adjust width as needed
                iframe = IFrame(html, width=200+20, height=100)  # Width includes a little extra for padding, height is as needed
                popup = folium.Popup(iframe, max_width=2650)  # max_width is set high to allow custom width to take effect
                folium.Marker(
                    location=(row['latitude'], row['longitude']),
                    popup=popup,
                    icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
                ).add_to(m)

        # Display the map
        m.save("templates/map.html")
        return 'map.html'
                       

    else:
        print("No optimal solution found.")

def routeFroToWarehouse(current_location, next_location, m, marker): 
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    now = datetime.now()  # Current time for routing purposes
    directions_result = gmaps.directions(current_location, next_location, mode="driving", departure_time=now)
    if directions_result:
        polyline = directions_result[0]['overview_polyline']['points']
        points = googlemaps.convert.decode_polyline(polyline)
        folium.PolyLine([(point['lat'], point['lng']) for point in points], color='blue', weight=2.5, opacity=1).add_to(m)
        if marker == 1:
            folium.Marker(
                location=(current_location[0], current_location[1]),
                popup='Source Warehouse',
                icon=folium.Icon(color='blue', prefix='fa')
            ).add_to(m)

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

def main(dataaa, vehicleNum, vehicleCap, sourceWarehouseCoord, destWarehouseCoord): 
    distance_duration = []
    print("##############################\n>> Main Program Started\n##############################")
    global m
    ##############################
    # Locating ports from warehouseSS 
    ##############################
    sourcePort=closestPort(PSAPORTDICT, sourceWarehouseCoord)
    destPort=closestPort(PSAPORTDICT, destWarehouseCoord)
    ##############################
    # Optimizing SEAROUTE:     
    ##############################
    print("##############################\n>> Optimising Sea Route\n##############################")   
    try:
        coord = [sourcePort, destPort]
        avoidHRA = False
        avoidSeca = False
        allowIceAreas = False
        blockAreas = []
        data = searoutes_query(coord)
        if not data:
            print("No route data available due to API limit. Exiting route generation.")
            return "Error: Rate limit exceeded for Searoutes API."
        line_coords, start_coord, end_coord, area_coords, area_names = area_data(data)
        m = create_map(line_coords)
        # m=create_clickable_map()

        plot_map(line_coords, start_coord, end_coord, area_coords, area_names,0)
        data_panama, data_suez=check_suez_panama(area_names,coord)    
        ## TODO: get_route_distance
        distance_duration=get_route_distance(data, data_panama, data_suez)
        
        # 1 route == normal
        # 2 routes == normal + one avoid panama/suez
        # 3 routes == normal + avoid panama + avoid suez 
    except:
        print(" >> No optimized Route found for SEAROUTE.")

    print("##############################\n>> Completed Sea Route\n##############################\n")
    
    print("##############################\n>> Optimising Train Route\n##############################")   
    try:
        # Load the data
        links_df = pd.read_csv('./rail_nodes/links_df.csv')
        b_df = pd.read_csv('./rail_nodes/b_df.csv')

        # Convert 'Inf' to GRB.INFINITY for unlimited capacities
        links_df['u_ij'] = links_df['u_ij'].replace({'Inf': GRB.INFINITY})

        # Get the city name for the given coordinates
        city = get_address(sourceWarehouseCoord).get('city')
        print(f"The coordinates {sourceWarehouseCoord} are in: {city}")

        # Duplicate for bidirectional
        bidirectional_links = pd.concat([links_df, links_df.rename(columns={'start node i': 'end node j', 'end node j': 'start node i'})])

        # Convert DataFrame to dictionary for easy parameter access
        c = {(row['start node i'], row['end node j']): row['c_ij'] for index, row in bidirectional_links.iterrows()}
        u = {(row['start node i'], row['end node j']): row['u_ij'] for index, row in bidirectional_links.iterrows()}
        b = {row['node i']: row['b_i'] for index, row in b_df.iterrows()}

        # Extract unique nodes
        nodes = sorted(set(b_df['node i']))

        # TODO: make 30 the volume instead
        # add demand and supply to node
        b_df.loc[b_df['node i'] == city, 'b_i'] = 30
        b_df.loc[b_df['node i'] == 'Qinzhou', 'b_i'] = -30.

        # Initialize model
        model = Model('mincost')

        # Extract the set of nodes
        nodes = list(set(bidirectional_links['start node i']).union(bidirectional_links['end node j']))

        # Update creation of decision variables to avoid duplicates
        x = model.addVars(bidirectional_links.apply(lambda row: (row['start node i'], row['end node j']), axis=1),
                        ub=[u_ij for u_ij in bidirectional_links['u_ij']],
                        name="x")

        # Correct the dictionary creation for costs and capacities
        c = {(row['start node i'], row['end node j']): row['c_ij'] for index, row in bidirectional_links.iterrows()}
        u = {(row['start node i'], row['end node j']): row['u_ij'] for index, row in bidirectional_links.iterrows()}

        # Supply/Demand for each node
        b = {row['node i']: row['b_i'] for index, row in b_df.iterrows()}

        # Objective: Minimize total cost
        model.setObjective(sum(c[i, j] * x[i, j] for i, j in x.keys()), GRB.MINIMIZE)

        # Flow conservation constraints
        for node in nodes:
            model.addConstr(
                sum(x.select(node, '*')) - sum(x.select('*', node)) == b.get(node, 0),
                name=f"flow_{node}")

        # Solve the model
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Optimal Solutions:")
            nodes_in_order = []
            for var in x.values():
                if var.X > 0:
                    print(f"{var.VarName}: {var.X}")
                    # Extract nodes from the variable name
                    var_name = var.VarName
                    # Assuming the format is x[city1,city2]
                    city1, city2 = var_name[var_name.find('[')+1:var_name.find(']')].split(',')
                    if not nodes_in_order:
                        nodes_in_order.append(city1)
                    nodes_in_order.append(city2)
            print(f"Total Cost: {model.ObjVal}")
            #distance in km
            distance_europe=model.ObjVal
        else:
            print("No optimal solution found.")

        # Add to folium map
        start_coords = [b_df.loc[b_df['node i'] == nodes_in_order[0], 'latitude'].values[0],
                        b_df.loc[b_df['node i'] == nodes_in_order[0], 'longitude'].values[0]]

        # Add markers and lines for each node in order
        for i in range(len(nodes_in_order)):
            node = nodes_in_order[i]
            lat = b_df.loc[b_df['node i'] == node, 'latitude'].values[0]
            lon = b_df.loc[b_df['node i'] == node, 'longitude'].values[0]
            folium.Marker([lat, lon], popup=node).add_to(m)
            if i > 0:
                prev_node = nodes_in_order[i-1]
                prev_lat = b_df.loc[b_df['node i'] == prev_node, 'latitude'].values[0]
                prev_lon = b_df.loc[b_df['node i'] == prev_node, 'longitude'].values[0]
                folium.PolyLine(locations=[[prev_lat, prev_lon], [lat, lon]], color='brown').add_to(m)
        routeFroToWarehouse(sourceWarehouseCoord, start_coords, m, 0)

        China_coords = [PSAPORTDICT['China']['PSA Qinzhou'],destPort]
        train_data = searoutes_query(China_coords)
        if not train_data:
            print("No route data available due to API limit. Exiting route generation.")
        train_line_coords, train_start_coord, train_end_coord, train_area_coords, train_area_names = area_data(train_data)
        plot_map(train_line_coords, train_start_coord, train_end_coord, train_area_coords, train_area_names,0)
        distance_duration_train=get_route_distance(train_data)
        total_train_distance=distance_duration_train[0][0] + distance_europe
        # assume 15 days for Europe to China since between 13-16 days, and 100km/h for the rest of the journey
        total_train_duration=distance_duration_train[0][1] + 15 + (distance_europe - 11179)/100/8.64e+7
        trainPath=[distance_europe,distance_duration_train[0][0],total_train_duration]
        distance_duration.append(trainPath)
    except:
        print(" >> No optimized Route found for SEAROUTE.")

    
    print("##############################\n>> Completed Train Route\n##############################\n")
    
    ##############################
    # Optimizing Warehouse route
    ##############################
    print("##############################\n>> Optimizing Warehouse Route\n##############################")
    try:
        # Configuring for our Optimization
        data_io = StringIO(dataaa)                 
        orders_df = pd.read_csv(data_io)                                        
        
        ## TODO: in the future, Vincenzo can control the Capacity of each vehicle carry
        depot_demand = 0   

        ## TODO: in the future, Vincenzo can control the number of vehicles: int
        K = vehicleNum

        ## TODO: in the future, Vincenzo can control the Vehicle capacity in m^3: int
        C = vehicleCap                
        
        delivery_route(orders_df, destWarehouseCoord, depot_demand, K, C, m)
        routeFroToWarehouse(sourceWarehouseCoord, sourcePort, m, 1)
        routeFroToWarehouse(destWarehouseCoord, destPort, m, 0)
        # Save the final map to HTML
        m.save("templates/map.html")
        
        # Return the map as HTML for direct embedding
        return distance_duration, m._repr_html_()
    except:
        print("Error occur in Warehouse route.")
    print("##############################\n>> Main Program Ended\n##############################")