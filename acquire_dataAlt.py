import requests
import json
import time
from datetime import datetime
import numpy as np


# ###########################################
# Parameters 
# ###########################################

Tscan       = 1  # Sampling period for data retrieval  from API
Rscan       = 10  # Scan radius around ref [nm] // api.airplanes.live limits to 250nm

# Latitude and longitude of radius-reference
EDIN_COORD  = [55.94843,-3.19658]
AMS_COORD   = [52.37205, 4.89048]
LND_COORD   = [51.51232, -0.11292]
PARIS_COORD = [48.84112, 2.33181]  
MNCH_COORD  = [48.13894, 11.57770]
GNV_COORD   = [46.20265, 6.14067]
MIL_COORD   = [45.46615, 9.18818]
TOUL_COORD  = [43.60182, 1.44124]
BCN_COORD   = [41.38613, 2.16909] 
MDR_COORD   = [40.41807, -3.69567]

COORDS = [
    EDIN_COORD,
    AMS_COORD,
    LND_COORD,
    PARIS_COORD,
    MNCH_COORD,
    GNV_COORD,
    MIL_COORD,
    TOUL_COORD,
    BCN_COORD,
    MDR_COORD,
    ]



API_URL     = "https://api.airplanes.live/v2/"

# ###########################################
# endpoint request config
# see fields in: https://airplanes.live/rest-api-adsb-data-field-descriptions/
# ###########################################


def fetch_and_store_data(coords):


    full_url = API_URL + "point/" + str(coords[0])+"/"+str(coords[1])+"/"+str(Rscan)
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
                "source_intgr": aircraft.get("sil"),
                "now": data.get("now"),
            }
            data_fmt.append(entry)
    
    return data_fmt

    
# #####################################
# MAIN 
# #####################################

all_data = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data_{timestamp}.json"
print(f"Starting data dump into {filename}")
ctr    = 0
Lcoord = np.shape(COORDS)[0]
try:
    while ctr<2:

        these_coords = COORDS[ctr % Lcoord] # np.remainder(ctr,Lcoord)
        thedata = fetch_and_store_data(these_coords)

        # all_data.append(the_data) # <<<<<<<<<<<<<<<<

        # Write to file
        with open(filename, 'a') as f:
            json.dump(thedata, f, indent="")        
            # json.dump(all_data, f, indent=2) # <<<<<<<<<<<<<<
                
        # print(f"Data added to {filename}")
        ctr += 1
        time.sleep(Tscan)


except KeyboardInterrupt:
    print("Data collection stopped.")



