#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install folium')
get_ipython().system('pip install altair')


# In[13]:


import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import folium as fo
import altair as alt


# In[14]:


data = pd.read_csv('J1_VIIRS_C2_SouthEast_Asia_24h.csv')
data = data[['latitude', 'longitude']]
data.head()


# In[15]:


n_clusters = 4250
kmeans = KMeans(n_clusters)
kmeans.fit(data)
cluster_labels = kmeans.labels_


data['cluster_labels'] = cluster_labels

silhouette_avg = silhouette_score(data, cluster_labels)
silhouette_avg


# In[16]:


latitudes = list(data.latitude)
longitudes = list(data.longitude)
cluster_labels = list(data.cluster_labels)


startpt = [12.5,101]


station_map = fo.Map(
    location = startpt, 
    zoom_start = 5) 

x = np.arange(n_clusters)
ys = [i + x + (i*x)**2 for i in range(n_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat,lng,cluster in zip(latitudes,longitudes,cluster_labels):
    fo.vector_layers.CircleMarker(
        location = [lat, lng], 
        color=rainbow[cluster-1],
        radius = 5,
        fill=True,
        fill_opacity=0.7
     ).add_to(station_map)
station_map


# In[ ]:




