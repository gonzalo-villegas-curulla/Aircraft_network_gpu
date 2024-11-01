import requests
import json
import time
from datetime import datetime
import numpy as np


# ###########################################
# Parameters 
# ###########################################

Tscan       = 1  # Sampling period for data retrieval  from API
Rscan       = 250  # Scan radius around ref [nm] // api.airplanes.live limits to 250nm

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

BREST_COORD    = [48.39758, -4.48612]
CORUNA_COORD   = [43.35378,-8.42859]
MALLRCA_COORD  = [39.61415,2.93885]
LAGOS_COORD    = [37.09462,-8.68878]
MALAGA_COORD   = [36.73145,-4.45168]
CASABLNC_COORD = [33.99802,-6.74599]
AGADIR_COORD   = [30.39699,-9.56286]
PALMAS_COORD   = [28.10348,-15.43305]

COORDS = [
    EDIN_COORD,
    AMS_COORD,
    LND_COORD,
    PARIS_COORD,
    BREST_COORD,
    MNCH_COORD,
    GNV_COORD,
    MIL_COORD,
    TOUL_COORD,
    CORUNA_COORD,
    BCN_COORD,
    MDR_COORD,
    MALLRCA_COORD,
    LAGOS_COORD,
    MALAGA_COORD,
    CASABLNC_COORD,
    AGADIR_COORD,
    PALMAS_COORD,
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
                # "type": aircraft.get("type"),
                "ground_speed": aircraft.get("gs"), 
                "true_speed": aircraft.get("tas"),
                "altitude": aircraft.get("alt_baro"),
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

# all_data = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data_{timestamp}.json"
print(f"Starting data dump into {filename}")
ctr    = 0
Lcoord = len(COORDS)

# For well-formed JSON format, we put [ at the beginnig and ] end of all the appended objects
with open(filename,'w') as f:
     f.write("[")




try:
    while True:

        # Loop through the set of coordinates at each iteration
        these_coords = COORDS[ctr % Lcoord] # np.remainder(ctr,Lcoord)
        thedata      = fetch_and_store_data(these_coords)      

        # Write to file
        with open(filename, 'a') as f:
            if ctr>0:
                 f.write(",\n")
            json.dump(thedata, f, indent="")        
                
        ctr += 1
        time.sleep(Tscan)

except KeyboardInterrupt:
    print("Data collection stopped.")
finally:
     # Pending JSON array formatting:
     with open(filename,'a') as f:
          f.write("]\n")


