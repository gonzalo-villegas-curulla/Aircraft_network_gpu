import requests
import json
import time
from datetime import datetime
# import numpy as np


# ###########################################
# Parameters 
# ###########################################

Tscan       = 1  # Sampling period for data retrieval  from API
Rscan       = 250  # Scan radius around ref [nm] // api.airplanes.live limits to 250nm

# Latitude and longitude of radius-reference
EDIN_COORD  = [55.94843,-3.19658]
AMS_COORD   = []
LND_COORD   = [51.51232, -0.11292]
PARIS_COORD = [48.84112, 2.33181]  
MNCH_COORD  = 
GNV_COORD = []
MIL_COORD = []
TOUL_COORD  = [43.60182,1.44124]
BCN_COORD   = [41.38613,2.16909] 
MDR_COORD



API_URL     = "https://api.airplanes.live/v2/"

# ###########################################
# endpoint request config
# see fields in: https://airplanes.live/rest-api-adsb-data-field-descriptions/
# ###########################################


def fetch_and_store_data():
    full_url = API_URL + "point/" + str(PARIS_COORD[0])+"/"+str(PARIS_COORD[1])+"/"+str(Rscan )
    # full_url = API_URL + "/mil" # For militar aircrafts
    response = requests.get(full_url)
    response.raise_for_status()
    data     = response.json()


    data_fmt = []
    for aircraft in data.get("ac", []):
            entry = {
                "hex": aircraft.get("hex"),
                "type": aircraft.get("type"),
                "altitude": aircraft.get("alt_baro"),
                "ground_speed": aircraft.get("gs"), 
                "true_speed": aircraft.get("tas"),
                "latitude": aircraft.get("lat"),
                "longitude": aircraft.get("lon"),
                "baro_rate": aircraft.get("baro_rate"),
            }
            data_fmt.append(entry)
    
    return data_fmt

    
# #####################################
# MAIN 
# #####################################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data_{timestamp}.json"
print(f"Starting data dump into {filename}")
try:
    while True:
        thedata = fetch_and_store_data()

        # Write the data to file
        with open(filename, 'a') as f:
            json.dump(thedata, f, indent=4)        
        # print(f"Data added to {filename}")

        time.sleep(Tscan)
except KeyboardInterrupt:
    print("Data collection stopped.")



