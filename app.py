#! /usr/bin/env python
"""
    This script is a Flask web application that provides an interface for optimizing routes between warehouses and destinations. 
    It includes a route to serve an HTML page and another route to handle route optimization requests. 
    The optimization results, including distance and duration, are displayed in a styled HTML format.
"""

from flask import Flask, jsonify, request, render_template, render_template_string
from flask_cors import CORS
import main

app = Flask(__name__)
CORS(app)  # Optional: Enable CORS for API if accessed from different origins

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('front.html')

def buildingBlock(distanceArr):         
    colorr = ["blue", "darkblue", "orange"]
    additional_html = """        
    <div class="container">
    """
    print(distanceArr)
    for count, eachDist in enumerate(distanceArr):
        if len(eachDist) == 2:
            additional_html += """
            <div class="box {color}">     
                <h3>Path: {color}</h3>           
                <div class="details">Distance: {distance:.3f} KM</div>
                <div class="details">Duration: {duration:.3f} Days</div>
            </div>
            """.format(
                color=colorr[count],                
                distance=eachDist[0],
                duration=eachDist[1]
            )
        if len(eachDist) == 3:
            additional_html += """
            <div class="box red">    
                <h3>Path: Red</h3>                             
                <div class="details">Distance travelled by Train: {distanceTrain:.3f} KM</div>
                <div class="details">Distance travelled by Ship: {distanceShip:.3f} KM</div>
                <div class="details">Total Distance travelled: {totalDistanceTravelled:.3f} KM</div>                
                <div class="details">Duration: {duration:.3f} Days</div>
            </div>
            """.format(                             
                distanceTrain=eachDist[0],
                distanceShip=eachDist[1],
                totalDistanceTravelled=eachDist[0] + eachDist[1],
                duration=eachDist[2]
            )
    additional_html += "</div>"
            
    return additional_html

@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    print(">> app.py:57 Optimize Route:")
    
    data = request.get_json()

    # Extract the coordinates and parameters from the received JSON
    sourceWarehouse = (data['sourceLat'], data['sourceLng'])
    destWarehouse = (data['destLat'], data['destLng'])
    # destinations = [(coord['address'], coord['lat'], coord['lng']) for coord in data['destinations']]    
    finalStr = "address,latitude,longitude,volume\n"
    for coord in data['destinations']:                        
        finalStr += '{}, {}, {}, 1\n'.format(coord['address'], coord['lat'], coord['lng'])                        
    

    distanceArr, map_file = main.main(finalStr, 1, 20, sourceWarehouse, destWarehouse)

    ### distance information together with map file
# Style definitions
    style_block = """
    <style>
        .container {
        display: flex;
        justify-content: space-around;
        padding: 20px;
        gap: 20px;
        }
        .box {
        width: 220px;
        padding: 20px;
        color: #333; /* Darker text for contrast */
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Soft shadow */
        background-color: #f9f9f9;
        text-align: center;
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease;
        }
        .box:hover {
        transform: translateY(-5px); /* Lift effect on hover */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* Shadow effect on hover */
        }
        .red {
        background-color: #d9534f;
        }
        .darkblue {
        background-color: #357abd;
        }
        .blue {
        background-color: #4a90e2;
        }
        .orange {
        background-color: #ffd966;
        }
        .box h3 {
        margin-top: 0;
        font-size: 1.4em;
        color: #555;
        }   
        .details {
            margin: 10px 0;
            font-size: 1.1em;
            color: #444;
            font-weight: 500;
        }
    </style>
    """
    additional_html = buildingBlock(distanceArr)
    complete_html = f"{style_block}{map_file}{additional_html}"    
    return render_template_string(complete_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
