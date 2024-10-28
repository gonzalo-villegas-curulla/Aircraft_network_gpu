import requests
import json
import time
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# ###########################################
# Parameters 
# ###########################################

Tscan       = 1  # Sampling period for data retrieval  from API
Rscan       = 5000  # Scan radius around Paris [km]
PARIS_COORD = (48.8566, 2.3522)  # Latitude and longitude of reference (Paris)
API_URL     = "https://api.airplanes.live/v2/icao/{icao}"  # Replace {icao} with actual aircraft ICAO if needed



# ###########################################
# endpoint request config
# see fields in: https://airplanes.live/rest-api-adsb-data-field-descriptions/
# ###########################################

# Compete as required 
endpoints = [
        "/mil"
        ]



# ###########################################
# Funcs
# ###########################################

# Calculate distance between two lat/lon points using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371.0  # Earth radius [km]
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a    = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c    = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ###########################################
# Time stamp for data storage
start_time = datetime.now().strftime("%Y%m%d_%H%M")
filename   = f"data_{start_time}.json"




# ###########################################
def fetch_and_store_data():

    data_list = [] 
    for endpoint in endpoints:
        try:
            full_url = API_URL + endpoint 
            response = requests.get(full_url)
            response.raise_for_status()
            data     = response.json()

            # If exists aircraft data 
            if "ac" in data:
                data_list.extend(data["ac"])
        except:
            print("Error fetching data from {url}: {e}")

    
    # Append data to the file with a timestamp
    with open(filename, "w") as current_file:
        # timestamped_data = {
        #     "timestamp": datetime.now().isoformat(),
        #     "aircrafts": filtered_data
        # }
        json.dump(data_list, current_file, indent=2)
        current_file.write("\n")

    print(f"Data fetched and stored at {datetime.now().isoformat()}")


# Continuous data retrieval loop until keyboard interruption
try:
    while True:
        fetch_and_store_data()
        time.sleep(Tscan)
except KeyboardInterrupt:
    print("Data collection stopped.")






























































































































