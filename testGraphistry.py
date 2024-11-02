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


# node2node distance: ON THE CPU ======================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


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
print(f"The compute of vectorized distances took {end-init:.2f} seconds")



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
print(f"The kernel computation of node2node dists and retrieve took {end-init:.2} seconds")

### THE VECTORIZED calc takes 420ms
### THE CUDA KERNEL takes 460ms


#  =============================== SORT NODES WITH ZIP and LOOP

distance_threshold = 50 # unitless for now
init = time.time()
within_threshold = (dist_MX <= distance_threshold)
i_indices, j_indices = np.where(np.triu(within_threshold, k=1))  # k=1 to avoid self-pairs and duplicates
edges = [(node_ids[i], node_ids[j], dist_MX[i, j]) for i, j in zip(i_indices, j_indices)]
end = time.time()
print(f"Edges with zip took {end-init:.2f} seconds")




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

# Define CUDA kernel to populate edges
# @cuda.jit
# def filter_edges(distances_matrix, threshold, edge_i, edge_j, edge_distances, edge_count):
#     i, j = cuda.grid(2)
#     if i < distances_matrix.shape[0] and j < distances_matrix.shape[1] and i < j:
#         if distances_matrix[i, j] <= threshold:
#             # Atomically increment edge count and get index for the edge
#             idx = cuda.atomic.add(edge_count, 0, 1)
#             edge_i[idx] = i
#             edge_j[idx] = j
#             edge_distances[idx] = distances_matrix[i, j]

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
print(f"The sorting of edges in kernel took {end-init:.2} seconds")





distance_threshold = 50 # unitless for now

init = time.time()
edges = [(node_ids[idx], node_ids[jdx], edge_data[idx,jdx].item())
         for idx in range(len(node_ids)) for jdx in range(idx+1, len(node_ids))
         if edge_data[idx,jdx] <= distance_threshold]
end = time.time() 
print(f"The populating of edge-data took {end-init:.2f} seconds")


edges_df = pd.DataFrame(edges, columns=['source','destination','distance'])
print(edges_df.head())

a = 2


# ===============================================
# ===============================================


# # Precompute node distances on the GPU
# Ldata = len(flattened_data)
# distances = np.empty([L,L])



# from numba import cuda
# @cuda.jit
# def hav_gpu():


#  Generate edges based on distance threshold
edge_list = []
distance_threshold = 50  # km

for (idx1, row1), (idx2, row2) in combinations(nodes_df.iterrows(), 2):
    dist = haversine(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude'])
    # if dist <= distance_threshold:
    edge_list.append({'source': row1['hex'], 'destination': row2['hex'], 'distance': dist})


edges_df = pd.DataFrame(edge_list)

# print(edges_df.head())
end = time.time()
print(f"This section took {end-init:.2f} s.")


# ==================== 
if True:
    init = time.time()
    # graphistry.register(api=3, protocol='https', server='hub.graphistry.com', username='YOUR_USERNAME', password='YOUR_PASSWORD')
    graphistry.register(api=3)

    # Bind and plot
    plot = graphistry.bind(
        source='source', destination='destination', node='hex'
    ).nodes(nodes_df).edges(edges_df).encode_point_color(
        'ground_speed', palette='Viridis', as_continuous=True
    ).encode_point_size(
        'altitude', as_continuous=True
    ).encode_x(
        'longitude'
    ).encode_y(
        'latitude'
    ).settings(url_params={'height': 800, 'play': 4000})

    end = time.time()
    print(f"This other section took {end-init:.2} seconds.")

    plot.plot()
# %%
