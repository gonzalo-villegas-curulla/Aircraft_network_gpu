# Scope

This project spun off the developments in network and graph representation and analysis of an [epidemioloy process](https://github.com/gonzalo-villegas-curulla/EpidemiologyProblem.git).

In the epidemiology model, the cells were represented by nodes of a graph and treated as a Gillespie process for two types of events (infection and recovery). In the present case, the network is constituted by aircrafts and airports, see a simple representation (without graph edges) below

![](assets/Simple01.jpeg)



# Methods

Data on aircraft location and other attributes is accessible by means of ADS-B signal and RTL-SDR (Realtek Software-Defined Radio) devices. One of the most extended projects is [Airplanes.Live](https://airplanes.live/api-guide/), crowd-sourced, open source and with a documented API. Further reading on RTL-SDR, documentation, projects and posts may be found in the [RTL-SDR Blog](https://www.rtl-sdr.com/). It's a good start to get to know the minimum setup hardware, portable options, and how to choose and configure your setup.


# Potential objectives
* Determine robustness or weakness of the network 
* Analyse cascaded events (e.g. delays, airspace buffer areas)
* Inspect the congestion (node degree, centrality, hubs) of main routes 
* Differentiate the behaviour of public airlines, cargo aircrafts and militar aircrafts


# Challenges
* A definition for nodes and interpretation of what roles do (1) moving aircrafts and (2) airports play
* Accelerate code to process data, ideally, under quasi-real-time requests
* Utilisation of sub-graphs
* Differentiate between stationary (zero baro-rate) and transient aircrafts (i.e. landing or taking off)


# Tools
* Networkx
* CUDA, pyCUDA, cugraph (minimum compute capability of 7.0 is required to run on GPU device)
* Graphistry
* JSON data bases

# Example showcase

Just under one minute of acquisition in the West of Europe (see Airplaines.live API limitations), the data is encoded into 2120 nodes having used an arbitrary adjacency criteria of 50 km threshold between nodes. The most salient hubs are found by visual inspection in the vicinity of London and Paris, followed by Amsterdam. One possible visualization is found below:

![2120 aircraft nodes visualizaed with graphistry v2.41.8](assets/Sample02.png)