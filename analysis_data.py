
# conda activate cugraphen
import time
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

import json
# from pprint import pprint
# import pandas as pd

import networkx as nx
import nx_cugraph as nxcg
import cugraph
import cudf
import graphistry # gpu acceleration of graph visualization

biginit = time.time()

# nx.betweenness_centrality(G, k=10,backend="cugraph")
#                   or
# $ NX_CUGRAPH_AUTOCONFIG=True python analysis_data.py 
#                   or
# import nx_cugraph as nxcg 
# \\....\\
# nxcg_G = nxcg.from_networkx(G)
# nx.betweenness_centrality(nxcg_G, k=1000)
#                   or 
# nxcg.betweenness_centrality(G, k=1000)


# ###################################################
# FUNCTIONS
# ###################################################

# Neglecting aircraft heights versus earth radius
# Distance between two point on the great circle:
# https://en.wikipedia.org/wiki/Haversine_formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# ###################################################
# LOAD data
# ###################################################

thefile = 'data_20241031_101252.json'
thefile = 'data_20241031_105813.json'

with open(thefile) as f:
    data = json.load(f)

# Access data:
# data[0][2]["hex"]

# JSON data Fields:
    # hex
    # type
    # ground_speed = w.r.t. ground [knots]
    # true_speed   = pointwise velocity [knots]
    # altitude       [foot]
    # latitude       [dec degrees]
    # longitude      [dec degrees]
    # baro_rate    = rate of change (vertical)[]
    # now          = time since 1 Jan 1970 @00:00:00 GMT [s]



# ###################################################
# Populate G
# ###################################################

G = nx.Graph()

loop_ctr     = 0
aircraft_ctr = 0
IDX = 0

print(f"Populating G...\n")
print(f"Adding nodes...")

data_ctr = 0
ctr = 0
for IDX in range(len(data)):
    for entry in data[IDX]:
 
        hex_code = entry.get("hex")
        lat = entry.get("latitude")
        lon = entry.get("longitude")
        if lat is not None and lon is not None:
            G.add_node(hex_code, latitude=lat, longitude=lon)
            aircraft_ctr+=1
            
        loop_ctr    += 1
print(f"(done)({len(G):d} nodes)\n")        


print(f"Adding edges...")
# Add edges (check node uniqueness)
for node1 in G.nodes:
    lat1, long1 = G.nodes[node1]["latitude"], G.nodes[node1]["longitude"]    
    for node2 in G.nodes:
        lat2, lon2 = G.nodes[node2]["latitude"], G.nodes[node2]["longitude"]
        distance = haversine(lat1, long1, lat2, lon2)
        if not G.has_edge(node1,node2):
            G.add_edge(node1, node2, length=distance)
print(f"(done)\n")

# print(f"Total aircraft processed: {aircraft_ctr}")
# print(f"Total nodes in graph: {len(G.nodes)}")
# print(f"Total edges in graph: {len(G.edges)}")


# ###################################################
#  MATHS 
# ###################################################


# Convert: NetworkX --> nx_cuGraph
nxcg_G = nxcg.from_networkx(G)

if False:
    init = time.time()
    BC_GPU = nxcg.betweenness_centrality(nxcg_G)
    end = time.time()
    print(f"Betweenness on GPU: {end - init:.2f} seconds")

    # init = time.time()
    # BC_CPU = nx.betweenness_centrality(G)
    # end = time.time()
    # print(f"Betweenness on CPU: {end - init:.2f} seconds")



# ###################################################
# VISUALISATION CPU 
# ###################################################


if False:
    print(f"\nStarting visualization\n")

    plt.figure(figsize=(10, 8))

    print(f"Extracting node positions...")
    init=time.time()
    pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes}
    end=time.time()
    print(f"(Done). {end-init:.2f} seconds\n")

    # Nodes ======================
    print(f"Drawing nodes...")
    init = time.time()
    # nx.draw(G, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=8)    
    # nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", font_size=8)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue", alpha=0.3)
    # nx.draw_networkx_edges(...)
    # nx.draw_edges(), .draw_networkx_labels(), .draw_networkx_edge_labels()
    end = time.time()    
    print(f"(Done). {end-init:.2f} seconds \n")

    
    # Get acquisition coordinates and scatter them
    exec(open("coordinates.py").read())
    plt.scatter(np.fliplr(COORDS)[:,0],np.fliplr(COORDS)[:,1],s=50,c="red")

    plt.xlabel(r"Longitude (decimal deg)")
    plt.ylabel(r"Latitude (decimal deg)")
    ax=plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid()
    

    # Edges =======================
    if False:
        print("Drawing edges...")
        init = time.time()
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        nx.draw_networkx_edges(G, pos)
        end = time.time()
        print(f"(Done). {end-init:.2f} seconds \n")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Aircraft Position Network")        



    # Edges Labels =======================

    if False:
        print(f"Getting edge lengths...")
        edge_labels = nx.get_edge_attributes(G, 'length')
        print(f"(Done)\n")
        print("Drawing edges...")
        init = time.time()
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        nx.draw_networkx_edge_labels(G, pos, edge_labels="", font_size=6)
        end = time.time()
        print(f"(Done). {end-init:.2f} seconds \n")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Aircraft Position Network")





bigend = time.time()
print(f"Total script runtime: {bigend-biginit} s.")
plt.show()


# ###################################################
# VISUALIZATION WITH GRAPHISTRY (GPU)
# ###################################################

# Set up Graphistry
# graphistry.register(api=3)  # Adjust this if your API version differs
# graphistry.authenticate()  # Ensure authentication if required
