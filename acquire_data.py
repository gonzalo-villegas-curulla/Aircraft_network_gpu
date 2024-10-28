import requests
import json
import time
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# ###########################################
Tscan       = 1  # Sampling period for data retrieval  from API
Rscan       = 5000  # Scan radius around Paris [km]
PARIS_COORD = (48.8566, 2.3522)  # Latitude and longitude of reference (Paris)
API_URL     = "https://api.airplanes.live/v2/icao/{icao}"  # Replace {icao} with actual aircraft ICAO if needed

# Calculate distance between two lat/lon points using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Create a time-stamped file for data storage
start_time = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"data_{start_time}.json"

def fetch_and_store_data():
    try:
        # This endpoint can be updated to pull data on all visible aircrafts globally.
        response = requests.get(API_URL.format(icao="all"))  # Adjust endpoint based on documentation if necessary
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        data = response.json()

        # Filter data based on distance from Paris
        filtered_data = []
        for ac_data in data.get("ac", []):
            last_pos = ac_data.get("lastPosition", {})
            lat, lon = last_pos.get("lat"), last_pos.get("lon")
            if lat is not None and lon is not None:
                distance = haversine(PARIS_COORD[0], PARIS_COORD[1], lat, lon)
                if distance <= Rscan:
                    filtered_data.append(ac_data)

        # Append data to the file with a timestamp
        with open(filename, "w") as file:
            timestamped_data = {
                "timestamp": datetime.now().isoformat(),
                "aircrafts": filtered_data
            }
            json.dump(timestamped_data, file)
            file.write("\n")

        print(f"Data fetched and stored at {datetime.now().isoformat()}")

    except requests.RequestException as e:
        print("Error retrieving data:", e)

# Continuous data retrieval loop
try:
    while True:
        fetch_and_store_data()
        time.sleep(Tscan)
except KeyboardInterrupt:
    print("Data collection stopped.")






























































































































