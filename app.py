import mlflow
import random
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
import json
from mlflow.client import MlflowClient
from shapely.geometry import MultiPoint, Point,box
from shapely.ops import voronoi_diagram
import plotly.express as px
from shapely.geometry import mapping
import plotly.graph_objects as go

load_dotenv()

DAGSHUB_PAT = os.getenv('DAGSHUB_PAT')

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_PAT
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_PAT

@st.cache_resource
def load_model_from_registry():
    mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/Urban-Rides-Demand-Prediction-Engine-via-Geo-Clustering.mlflow')

    client = MlflowClient()

    registered_model_name = "Taxi_Demand_Prediction_Model"
    latest_version = client.get_model_version_by_alias(registered_model_name, alias='Staging')

    model_uri = f'models:/{registered_model_name}/{latest_version.version}'
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model_from_registry()

data_path = 'https://raw.githubusercontent.com/akshatsharma2407/Car-Data-API/refs/heads/master/test.csv'
df = pd.read_csv(data_path, parse_dates=['tpep_pickup_datetime']).set_index('tpep_pickup_datetime')

with open('coordinates.json', 'r') as f:
    cluster_centroids = json.load(f)


def get_ordered_voronoi_polygons(centroids_lat_lon, nyc_box_coords):
    points_list = [Point(c[0], c[1]) for _, c in centroids_lat_lon.items()]
    points_geom = MultiPoint(points_list)

    nyc_boundary = box(nyc_box_coords[1], nyc_box_coords[0], 
                       nyc_box_coords[3], nyc_box_coords[2])

    regions = voronoi_diagram(points_geom, envelope=nyc_boundary)

    ordered_polygons = []

    for pt in points_list:
        found_poly = None

        for region in regions.geoms:
            if region.contains(pt) or region.intersects(pt):
                found_poly = region
                break
        
        if found_poly:
            clipped = found_poly.intersection(nyc_boundary)
            ordered_polygons.append(clipped)
        else:
            ordered_polygons.append(None) 

    return ordered_polygons


def convert_polys_to_geojson(polygons):
    features = []
    for i, poly in enumerate(polygons):
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "id": i,                    
            "properties": {"cluster_id": i}
        })
    return {"type": "FeatureCollection", "features": features}



def haversine_distance(lat1:float, lon1:float, lat2:float, lon2:float):
        
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
            dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
        
    earth_radius = 3958.8
    distance = earth_radius * c
    return distance


def render_project_description():
    
    st.markdown("""
    ### 🎯 The Business Objective
    In the ride-hailing ecosystem, our growth is intrinsically linked to our driver partners. **When they earn more, we earn more.**
    
    The biggest enemy of profitability is **Idle Time**. If a driver is waiting without a ride, they are burning fuel without generating revenue. To solve this, we built a system that predicts demand for the **next 15-minute interval**, allowing us to guide drivers to high-demand areas *before* the surge happens. This project is build over the NYC Yellow Taxi trip dataset, purpose of taking this dataset is to handle the scale **(5GB of dataset)**, used efficient coding practice and library like dask to build this project on a **machine with only 8GB of RAM**
   
    ---
    
    ### 📐 Spatial Engineering & Clustering
    We moved beyond static administrative boundaries (like zip codes) to create dynamic, data-driven operational zones using **K-Means Clustering**.
    
    **The "Sweet Spot" Constraint:**
    We didn't just cluster the data; we optimized for operational reality. We ran K-Means for various $K$ values to ensure the majority of clusters span a **1–1.5 mile radius**.
    
    * **💰 Cost Efficiency:** Limits "dead mileage" so drivers don't burn excess fuel reaching a zone.
    * **⏱️ Time Viability:** In NYC traffic, 1.5 miles is reachable within **~15 minutes**, ensuring drivers arrive exactly when the demand is needed.
    * **Result:** Using the **Haversine formula** for distance validation, we identified the optimal number of zones ($K$) as **30**.
    
    ---
    
    ### 🧠 Predictive Modeling (XGBoost)
    We treated this as a time-series forecasting problem using the first 3 months of NYC data (Jan-Feb for training, Mar for testing).
    
    * **Algorithm:** **XGBoost Regressor** trained on temporal lag features.
    * **Custom Loss Function:** Instead of standard MSE, I implemented a custom **SMAPE (Symmetric Mean Absolute Percentage Error)** objective. This ensures the model is robust against the high variance in real-world demand.
    
    ### 📱 The Solution
    This dashboard provides real-time actionable intelligence:
    1.  **Macro View:** A heat map of demand across the entire city.
    2.  **Micro View:** Specific predictions for **nearby clusters**, helping drivers make quick, profitable decisions.
    """)


min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70
nyc_bbox = [min_latitude, min_longitude, max_latitude, max_longitude]

polygons = get_ordered_voronoi_polygons(cluster_centroids, nyc_bbox)

nyc_geojson = convert_polys_to_geojson(polygons)


map_type = st.sidebar.radio(label='Select the type of Map', options=['Home', 'Complete NYC Map', 'Only for Neighborhood Regions', "About Me"], index=0)

if map_type == "Home":
    st.title("🚖 Urban Fleet Equilibrium Engine via Dynamic Geo-Clustering")
    render_project_description()

elif map_type == "Complete NYC Map" or map_type ==  'Only for Neighborhood Regions':
    if map_type == "Complete NYC Map":
        st.title("Demand Over Entire NYC map") 
    else :
        st.title("Demand Over Neighbour Regions")

    date = st.date_input("Select the date", value=None, min_value=dt.date(year=2016, month=3, day=1),
                max_value=dt.date(year=2016, month=3, day=31))
    time = st.time_input("Select the time", value=None)

    st.write('**Date:**', date)
    st.write('**current time**', time)


    if date and time:
        delta = dt.timedelta(minutes=15)
        next_interval = dt.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute
        ) + delta

        index = pd.Timestamp(f"{date} {next_interval.time()}")

        st.write("Demand for : ", next_interval)

        rows = df.loc[index].drop(columns='total_pickups')
        
        predictions = {}

        for i in rows.values:
            one_row = pd.DataFrame(data=[i], columns=rows.columns)
            one_row = one_row.astype(rows.dtypes)
            x = np.expm1(model.predict(one_row))
            predictions[i[0]] = np.round(x[0])
            
        demand_df = pd.DataFrame(data={'cluster_id': predictions.keys(), 'demand_level': predictions.values()})


        if map_type == 'Complete NYC Map':

            fig = px.choropleth_mapbox(
                demand_df,
                geojson=nyc_geojson,
                locations='cluster_id',       
                color='demand_level',        
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                center={"lat": 40.725, "lon": -73.875},
                zoom=10,
                opacity=0.4         
            )
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)
        
        else:
            st.write('taking a random location on map to simulate like fetching the live location!')
            few_cordinate = [[40.730610, -73.907042],[40.89150152352015, -73.9140293789116], [40.78733351539776, -73.95962405788946], [40.64191315310556, -73.76557878639855], [40.82213475260872, -73.98320269773465]]
            curr_cordinate = few_cordinate[random.randint(0, len(few_cordinate)-1)]
            distances = {}
            for key,val in cluster_centroids.items():
                distance = haversine_distance(curr_cordinate[0], curr_cordinate[1], val[1], val[0])
                distances[key] = distance
            neighbour_centroids = dict(sorted(distances.items(), key=lambda item: item[1])).keys()
            neighbour_centroids = list(neighbour_centroids)[:8]
            neighbour_centroids = [int(i) for i in neighbour_centroids]
            
            for i,j in predictions.items():
                if int(i) not in neighbour_centroids:
                    predictions[i] = 0
            
            demand_df = demand_df[demand_df['cluster_id'].isin(neighbour_centroids)]

            fig = px.choropleth_mapbox(
                demand_df,
                geojson=nyc_geojson,
                locations='cluster_id',       
                color='demand_level',        
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                center={"lat": 40.725, "lon": -73.875},
                zoom=10,
                opacity=0.4         
            )

            fig.add_trace(go.Scattermapbox(
            lat=[curr_cordinate[0]],
            lon=[curr_cordinate[1]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color='red',    
                opacity=1.0      
            ),
            text=['My Location'], 
            name='Current Location' 
            ))

            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)

else:
    st.markdown("""
    **Hi, I am Akshat Sharma.** 👋

    I am a Data Science practitioner, passionate about building intelligent systems that solve real-world problems. With a background in **Data Science**, I specialize in transforming raw data into actionable insights.

    My work focuses on:
    * **Predictive Modeling:** Not just code, but understanding the Maths behind ML / DL algo.
    * **Spatial Analytics:** applying geometric clustering (like K-Means & Voronoi) to optimize logistics and urban mobility.
    * **End-to-End Development:** Build some End to End project which you can checkout on my Portfolio.

    [**Connect with me on LinkedIn**](https://www.linkedin.com/in/akshat-sharma-89635631b/)
                
    [**Connect with me on Github**](https://github.com/akshatsharma2407/)
                
    [**My Portfolio**](https://akshatsharma2407.github.io/my_portfolio/)
    """)