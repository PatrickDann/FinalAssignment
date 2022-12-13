#!/usr/bin/env python
# coding: utf-8

# # Map Help for Final Project 

# In this project you may be tempted to create a map for your analysis. Well, things are not always straight forward so we are creating this additional ressource to help you if you are being ambitious. 

# In[1]:


# Importing in your required libraries
import pandas as pd
import altair as alt
alt.data_transformers.enable('default', max_rows=1000000)
import json


# Remember that when you export your notebook to an html file, comment out the line `alt.data_transformers.enable('data_server')` in order for the visualizations to output. 

# Let's bring in the data. For reference, this data is a subset of the original data available on [The Vancouver Data Portal](https://opendata.vancouver.ca/explore/dataset/street-trees/information/). The data that we have given you was adapted from the json file and wrangled slightly so that it's easy to use for geographical visualizations. 

# In[2]:


df = pd.read_csv('vancouver_trees.csv')


# In[3]:


df.head()


# Now we can use this data and make many different visualization (with additional wrangling) but this resource is here to help explain how we will make maps using this data. 

# Since Altair does not make Vancouver easy to locate on the global map and there is no projection for Canada like there is for the United states, we've made the geojson for Vancouver and it's neighbourhoods available through a url. This was obtain from the [Vancouver Data Portal](https://opendata.vancouver.ca/explore/dataset/local-area-boundary/map/) once again. 

# To make a base map of Vancouver we use the geojson url saved in `url_geojson`. 

# In[4]:


url_geojson = 'https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/local-area-boundary.geojson'


# Next, we must format it in a Topo json format which we convert using `alt.Data()`.

# In[5]:


data_geojson_remote = alt.Data(url=url_geojson, format=alt.DataFormat(property='features',type='json'))

data_geojson_remote


# We can then make our base Vancouver Altair map using the `data_geojson_remote` object as we've made maps in the past except this time we need to use an `identity` `type` and we need to `reflectY=True`.  Without this second argument our map of Vancouver is upside down. 

# In[6]:


vancouver_map = alt.Chart(data_geojson_remote).mark_geoshape(
    color = 'gray', opacity= 0.5, stroke='white').encode(
).project(type='identity', reflectY=True)

vancouver_map


# Nice, we have a base map of Vancouver! 🎉 
# 
# Now all we have to do is combine this with some of our tree data. 
# 
# Let's plot the median diameter of the tree trunks for each neighbourhood. 
# 
# I'm going to rename `neighbourhood_name` to `name` since that's what's it's called in the geojson url and we need to connect the two dataframes using the function `transform_lookup()`.
# 
# I'm also going to select the median latitude and longitude columns for each neighbourhood as I'm going to make a point map using these coordinates after. 

# In[7]:


median_df = df.groupby('neighbourhood_name'
                      ).median().reset_index(
).rename(columns={'neighbourhood_name':'name'})[['name',
                                                 'diameter', 
                                                 'latitude', 
                                                 'longitude']]
median_df


# This now gives us the median lat and long coordinates as well as tree trunk diameter per Vancouver neighbourhood. 

# Now we link the shape file with the median tree dataframe using lookups like we learned in Module 6. 
# 
# The neighbourhood is stored in the properties field, which we can access using `properties.name`. 
# 
# We then grab the `diameter` and `name` column from `median_df` using `LookupData()`. 
# 
# We color the neighbourhoods based on trunk `diameter` size. 

# In[8]:


alt.Chart(data_geojson_remote).mark_geoshape().transform_lookup(
    lookup='properties.name',
    from_=alt.LookupData(median_df, 'name', ['diameter', 'name'])).encode(
    color='diameter:Q',
    tooltip='name:N').project(type='identity', reflectY=True)


# Look at that! We have a chloropleth map! 
# 
# We learned that these can sometimes be a bit deceiving, so we can instead use point size to show diameter size instead. 
# 

# In[9]:


points = alt.Chart(median_df).mark_circle().encode(
    longitude='longitude',
    latitude='latitude',
    size='diameter:Q',
    color = 'diameter:Q',
    tooltip='name').project(type= 'identity', reflectY=True)

points


# And overlay it on our base Vancouver map. 

# In[10]:


(vancouver_map + points).configure_view(stroke=None)


# In addition we could plot all the trees in the dataset using the latitude and longitude of each row/tree in the full dataframe. 

# In[11]:


points = alt.Chart(df).mark_circle(size=1, color='green').encode(
    longitude='longitude',
    latitude='latitude').project(type= 'identity', reflectY=True)

(vancouver_map + points).configure_view(stroke=None)


# We've also provided you a geojson url file for each neighbourhood individually. 
# 
# - [Arbutus Ridge](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/arbutus_ridge.geojson)
# - [Downtown](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/downtown.geojson)
# - [Dunbar-Southlands](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/dunbar_southlands.geojson)
# - [Fairview](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/fairview.geojson)
# - [Grandview-Woodland](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/grandview_woodland.geojson)
# - [Hastings-Sunrise](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/hastings_sunrise.geojson)
# - [Kensington-Cedar Cottage](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/kensington_cedar_cottage.geojson)
# - [Kerrisdale](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/kerrisdale.geojson)
# - [Killarney](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/killarney.geojson)
# - [Kitsilano](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/kitsilano.geojson)
# - [Marpole](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/marpole.geojson)
# - [Mount Pleasant](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/mount_pleasant.geojson)
# - [Oakridge](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/oakridge.geojson)
# - [Renfrew-Collingwood](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/renfrew_collingwood.geojson)
# - [Riley Park](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/riley_park.geojson)
# - [Shaughnessy](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/shaughnessy.geojson)
# - [South Cambie](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/south_cambie.geojson)
# - [Strathcona](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/strathcona.geojson)
# - [Sunset](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/sunset.geojson)
# - [Victoria-Fraserview](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/victoria_fraserview.geojson)
# - [West End](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/west_end.geojson)
# - [West Point Grey](https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/west_point_grey.geojson)
# 

# This will give you the ability to plot an individual neighbourhood if you wish. 

# In[12]:


url_geojson_killarney = 'https://raw.githubusercontent.com/UBC-MDS/exploratory-data-viz/main/data/vancouver_neighbourhoods/killarney.geojson'


# In[13]:


data_geojson_remote_kil = alt.Data(url=url_geojson_killarney, format=alt.DataFormat(property='features',type='json'))


# In[14]:


killarney_map = alt.Chart(data_geojson_remote_kil).mark_geoshape(
    color = 'gray', opacity= 0.5, stroke='white').encode(
).project(type='identity', reflectY=True)

killarney_map


# In[17]:


df_kil = df[df['neighbourhood_name'] == 'Killarney']
df_kil.head()


# In[18]:


points_kil = alt.Chart(df_kil).mark_circle(size=5, color='green').encode(
    longitude='longitude',
    latitude='latitude').project(type= 'identity', reflectY=True)

points_kil


# In[19]:


(killarney_map + points_kil).configure_view(stroke=None)

