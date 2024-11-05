import requests
import json
import time
from datetime import datetime
import numpy as np


# ###########################################
# Parameters 
# ###########################################

Tscan       = 1    # Sampling period for data retrieval  from API
Rscan       = 250  # Scan radius around ref [nm] // api.airplanes.live limits to 250nm

# Get acquisition points' coordinates from external file
exec(open("coordinates.py").read())


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


