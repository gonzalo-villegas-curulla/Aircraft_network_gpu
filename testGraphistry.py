import json
import pandas as pd
from itertools import combinations
import math 
import graphistry
import time
import numpy as np
import cupy as cp
from numba import cuda

# Load the JSON data
with open('data_20241031_101252.json') as f:
    data = json.load(f)

# Flatten JSON data into a list of dictionaries
flattened_data = [entry for sublist in data for entry in sublist]

# Create nodes DataFrame
nodes_df = pd.DataFrame(flattened_data)

# Check data
# print(nodes_df.head())
init = time.time()

#####################################################################
#####################################################################

# node2node distance: VECTORIZED =======================================
# Format to cupy
latitudes  = cp.asarray(nodes_df['latitude'].values)
longitudes = cp.asarray(nodes_df['longitude'].values)
node_ids   = nodes_df['hex'].values


# Matrix (shorthand "MX") to store distances
edge_data = cp.zeros((len(latitudes), len(latitudes)), dtype=cp.float32)

R = cp.float32(6371.0)

# Vectorized haversine function for CuPy
def haversine_matrix(latitudes, longitudes):
    lat1 = latitudes[:, cp.newaxis]
    lon1 = longitudes[:, cp.newaxis]
    lat2 = latitudes[cp.newaxis, :]
    lon2 = longitudes[cp.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon / 2) ** 2
    c = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1 - a))
    return R * c
# Calculate pairwise distances
init = time.time()
edge_data = haversine_matrix(latitudes, longitudes)
end = time.time()
print(f"DISTANCES node2node: vectorized took {end-init:.2f} seconds")



# node2node distance: CUDA KERNEL ===============================
latitudes  = cp.asarray([math.radians(lat) for lat in nodes_df['latitude'].values])
longitudes = cp.asarray([math.radians(lon) for lon in nodes_df['longitude'].values])
node_ids   = nodes_df['hex'].values

dev_latitudes  = cuda.to_device(latitudes)
dev_longitudes = cuda.to_device(longitudes)
N_nodes        = len(latitudes)

dist_MX     = cp.zeros((N_nodes, N_nodes), dtype=cp.float32)
dev_dist_MX = cuda.to_device(dist_MX)

@cuda.jit
def hav_kernel(latitudes, longitudes, distances):
    idx, jdx = cuda.grid(2)
    if idx< distances.shape[0] and jdx<distances.shape[1] and idx<jdx:
        dlat = latitudes[jdx]- latitudes[idx]
        dlon = longitudes[jdx]-longitudes[idx]
        factor = cp.float32(2.0)
        one = cp.float32(1.0)
        a = math.sin(dlat/factor)**factor   +   math.cos(latitudes[idx]) * math.cos(latitudes[jdx]) * math.sin(dlon/factor)**factor
        c = factor * math.atan2( math.sqrt(a), math.sqrt(one -a))
        distances[idx,jdx] = R*c

bsize   = (16,16)
gsize_x = int(np.ceil(N_nodes / bsize[0]))
gsize_y = int(np.ceil(N_nodes/bsize[1]))
gsize   = (gsize_x, gsize_y)

init = time.time()
# Launch kernel
hav_kernel[gsize,bsize](dev_latitudes, dev_longitudes, dev_dist_MX)
# retrieve to host (is this necessary?)
dist_MX = dev_dist_MX.copy_to_host()
end = time.time()
print(f"DISTANCES node2node: kernel launch and to_host {end-init:.2} seconds")


#####################################################################
#####################################################################


#  =============================== SORT NODES WITH ZIP and LOOP

distance_threshold = 50 # unitless for now
init = time.time()
within_threshold = (dist_MX <= distance_threshold)
i_indices, j_indices = np.where(np.triu(within_threshold, k=1))  # k=1 to avoid self-pairs and duplicates
edges = [(node_ids[i], node_ids[j], dist_MX[i, j]) for i, j in zip(i_indices, j_indices)]
end = time.time()
print(f"EDGES SORTING with zip: {end-init:.2f} seconds")




#  =============================== SORT NODES KERNEL

num_nodes = dev_dist_MX.shape[0]
distance_threshold = np.float32(50.0)  # km, ensure float32 type for CUDA

# Pre-allocate arrays for storing the edges data
max_edges = num_nodes * (num_nodes - 1) // 2  # Maximum number of edges (upper triangle)
dev_edge_i = cuda.device_array(max_edges, dtype=np.int32)
dev_edge_j = cuda.device_array(max_edges, dtype=np.int32)
dev_edge_distances = cuda.device_array(max_edges, dtype=np.float32)
dev_edge_count = cuda.device_array(1, dtype=np.int32)  # To keep track of edge count

# Ensure distances_matrix is float32 before copying to the device
distances_matrix_host = dist_MX.astype(np.float32)
dev_distances_matrix = cuda.to_device(distances_matrix_host)



#  ***
from numba import float32, int32 
@cuda.jit
def filter_edges(distances_matrix, threshold, edge_i, edge_j, edge_distances, edge_count):
    # Shared memory for compacted edges (reduces atomic operations)
    sh_edge_i = cuda.shared.array(shape=(1024,), dtype=int32)
    sh_edge_j = cuda.shared.array(shape=(1024,), dtype=int32)
    sh_edge_distances = cuda.shared.array(shape=(1024,), dtype=float32)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bdx = cuda.blockDim.x

    # Linear index from grid dimensions
    i = by * bdx + tx
    j = bx * bdx + tx + 1  # Offset by 1 to avoid self-loops and duplicates

    # Initialize local counter for edges within threshold
    edge_counter = 0

    if i < distances_matrix.shape[0] and j < distances_matrix.shape[1] and i < j:
        dist = distances_matrix[i, j]
        
        if dist <= threshold:
            # Populate shared memory for compacted edges
            sh_edge_i[tx] = i
            sh_edge_j[tx] = j
            sh_edge_distances[tx] = dist
            cuda.syncthreads()
            
            # Copy shared memory to global memory (avoiding atomics here)
            edge_idx = cuda.atomic.add(edge_count, 0, 1)
            edge_i[edge_idx] = sh_edge_i[tx]
            edge_j[edge_idx] = sh_edge_j[tx]
            edge_distances[edge_idx] = sh_edge_distances[tx]

# Define grid size and block size
threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(num_nodes / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(num_nodes / threads_per_block[1]))

# Initialize the edge_count to 0
dev_edge_count[0] = 0

init = time.time()
# Run the kernel
filter_edges[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
    dev_distances_matrix, distance_threshold, dev_edge_i, dev_edge_j, dev_edge_distances, dev_edge_count
)

# Copy results back to the host
edge_i_host = dev_edge_i.copy_to_host()[:dev_edge_count[0]]
edge_j_host = dev_edge_j.copy_to_host()[:dev_edge_count[0]]
edge_distances_host = dev_edge_distances.copy_to_host()[:dev_edge_count[0]]

# Combine results
edges = list(zip(edge_i_host, edge_j_host, edge_distances_host))

end = time.time()
print(f"EDGES SORTING: kernel launch and to_host {end-init:.2} seconds")



#####################################################################
#####################################################################

init = time.time()
# Filter edges by distance threshold
distance_threshold = 50.0  # Adjust as needed
within_threshold = (dist_MX <= distance_threshold)
i_indices, j_indices = cp.where(cp.triu(within_threshold, k=1))
end = time.time()
print(f"preop1 {end-init:.2} seconds")


init = time.time()
# Convert indices and distances to host
i_indices_host = cp.asnumpy(i_indices)
j_indices_host = cp.asnumpy(j_indices)
# distances_host = cp.asnumpy(dist_MX[i_indices, j_indices])
distances_host = cp.asnumpy(dist_MX[i_indices_host, j_indices_host])
end = time.time()
print(f"Preop2 {end-init:.2} seconds")

init = time.time()
# Create edges DataFrame for Graphistry
edges_df = pd.DataFrame({
    'source': node_ids[i_indices_host],
    'destination': node_ids[j_indices_host],
    'distance': distances_host
})
end = time.time()
print(f"Preop3 {end-init:.2} seconds")


# ==================== 
if False:
    import matplotlib.pyplot as plt

    viridis_palette = [plt.cm.viridis(i) for i in range(256)]
    color_palette = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, _ in viridis_palette]

    graphistry.register(api=3, protocol='https', server='hub.graphistry.com', username='XXXXXXXXXXXXX', password='XXXXXXXXXXXX')

    # Bind and plot
    # plot = graphistry.bind(
    #     source='source', destination='destination', node='node_id'
    # ).nodes(nodes_df).edges(edges_df).encode_point_color(
    #     'ground_speed', palette=color_palette, as_continuous=True
    # ).encode_point_size(
    #     'altitude'
    # # ).encode_x(
    # #     'longitude'
    # # ).encode_y(
    # #     'latitude'
    # ).settings(url_params={'height': 800, 'play': 4000})

    plot = graphistry.nodes(nodes_df,'src','dst').edges(edges_df,'src','dst').settings(url_params={'height': 800, 'play': 4000})
    

    plot.plot()




#  =================================================
if False:
    import networkx as nx
    import pandas as pd
    import plotly.graph_objects as go
    from dash import Dash, dcc, html

    # Create a NetworkX graph from the nodes and edges data
    G = nx.Graph()

    # Add nodes with position and custom labels
    for _, row in nodes_df.iterrows():
        G.add_node(row['hex'], pos=(row['longitude'], row['latitude']))

    # Add edges with weights
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=row['distance'])

    # Get positions for each node
    positions = nx.spring_layout(G, seed=42)  # Can use any layout; spring_layout works well generally

    # Extract X, Y coordinates for nodes
    x_nodes = [positions[node][0] for node in G.nodes]
    y_nodes = [positions[node][1] for node in G.nodes]

    # Extract edges for X and Y coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x += [x0, x1, None]  # None creates a break in line segments for individual edges
        edge_y += [y0, y1, None]

    # Create the Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Interactive Network Graph"),
        dcc.Graph(id='network-graph',
                figure={
                    'data': [
                        # Edges as line shapes
                        go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines'
                        ),
                        # Nodes as scatter points
                        go.Scatter(
                            x=x_nodes, y=y_nodes,
                            mode='markers+text',
                            marker=dict(
                                color='SkyBlue', size=10, line=dict(width=1)
                            ),
                            text=[str(node) for node in G.nodes],  # Display node IDs
                            hoverinfo='text',
                        )
                    ],
                    'layout': go.Layout(
                        title='Network Graph Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                })
    ])

    # Run the Dash app
    if __name__ == '__main__':
        app.run_server(debug=True)