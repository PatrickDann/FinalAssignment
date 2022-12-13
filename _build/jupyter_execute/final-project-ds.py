#!/usr/bin/env python
# coding: utf-8

# # Disney Movie Data Analysis
# By Patrick Dann

# ## Introduction
# 
# Back to [Intro](intro.md).

# ## Question of Intrest 
# 
# In this analysis I will see which disney director has the highest average gross earnings for the Disney films they directed after inflation. There are many factors which can influence the gross earnings of a film. By looking at the average earning of a directors movies you look at other factors which may have influenced a movies earnings to  be below or above average for a director. 

# ## Dataset Description 
# 
# The Disney Data set has 5 tables: disney-characters.csv, disney-director.csv, disney-voice-actors.csv, disney_revenue_1991-2016.csv, disney_movies_total_gross.csv. They obtained from: https://data.world/kgarrett/disney-character-success-00-16 
# 
# I will examine the disney-director.csv and disney_movies_total_gross.csv files. 
# 

# ## Methods and Results
# 
# I will import the disney_movies_total_gross and disney-director tables since I want to look at disney movies gross by director.

# In[1]:


import altair as alt
import pandas as pd 

# Import all required files 
movies_total_gross = pd.read_csv(r'C:\Users\ichir\Documents\final-assignment/disney_movies_total_gross.csv')
disney_directors = pd.read_csv(r'C:\Users\ichir\Documents\final-assignment\disney-director.csv')


# In[2]:


movies_total_gross.head()


# Table 1. Disney movies Total and adjusted gross

# The movies_total_gross dataset has 6 columns; movie_title, release_date, genre, MPAA_rating, total_gross, and inflation_adjusted_gross.

# In[3]:


disney_directors.head()


# Table 2. Disney Movie Directors

# The disney_directos dataset had 2 columns; name and director. 

# In[4]:


movies_total_gross.info()


# There are 579 rows and 6 columns for the movies_total_gross dataset. 

# In[5]:


disney_directors.info()


# The disney_directors dataset has 56 rows and 2 columns.

# I want to visualize which genre of disney movie has the highest average gross. I will use the inflation_adjusted_gross column from the movies_total_gross dataset since these values account for inflation and many of the disney movies were released many years ago. 
# 
# First I have to convert the dtype of the inflation_adjusted_gross from object to int.

# In[6]:


# Remove commas and symbols from inflation_adjusted_gross so we can convert str to numerical
movies_total_gross['inflation_adjusted_gross'] = movies_total_gross['inflation_adjusted_gross'].str.replace(',','')
movies_total_gross['inflation_adjusted_gross'] = movies_total_gross['inflation_adjusted_gross'].str.replace('$','')


# In[7]:


# Change inflation_adjusted_gross to numerical dtype so we can average it
movies_total_gross['inflation_adjusted_gross'] = pd.to_numeric(movies_total_gross['inflation_adjusted_gross'])


# Now I will group the data by genre and average the gross after adjusted for inflation.

# In[8]:


# Group by genre and determine average gross
genre_avg_gross = pd.DataFrame(movies_total_gross.groupby('genre')['inflation_adjusted_gross'].mean().sort_values(ascending = False))

genre_avg_gross = genre_avg_gross.reset_index()

genre_avg_gross


# Table 3. Average Adjusted gross of disney movies sorted by genre 

# Now I am ready to visualize the data. I can make a bar plot to see which genre of disney movie has the highest average gross. 

# In[9]:


# Use altair to generate a bar plot
genre_gross_plot = (
    alt.Chart(genre_avg_gross, width=500, height=300)
    .mark_bar()
    .encode(
        x=alt.X("genre:N", sort='-y', title="Genre"),
        y=alt.Y("inflation_adjusted_gross:Q", title="Average Adjusted Gross"),
    )
    .properties(title="Average Adjusted Gross of Disney Films by Genre")
)
genre_gross_plot


# Figure 1. Bar graph of Average Adjusted Gross of Disney Films By Genere 

# Here we can see that disney musicals have a significantly higher gross than other disney movie genres. 

# Now I will combine the disney_directors and movies_total_gross datasets, group the new combined dataset by director and average the inflation_adjusted_gross with a custom function.

# In[10]:


# Import the custom function
from function import merge_group_avg


# In[11]:


# Running the custom funtion on the datasets
director_gross = merge_group_avg(disney_directors, movies_total_gross, 'director', 'inflation_adjusted_gross')
director_gross


# Table 4. Disney Directors Average Adjusted gross of their Disney Movies

# Now I have a dataset with the disney directors and their average gross from their disneys movies after inflation. 
# 
# I can plot this to visualize this with a bar plot

# In[12]:


# Visualize the Disney directors with the highest average gross from their Disney films
director_gross_plot = (
    alt.Chart(director_gross, width=500, height=300)
    .mark_bar()
    .encode(
        x=alt.X("director:N", sort='-y', title="Director"),
        y=alt.Y("inflation_adjusted_gross:Q", title="Average Adjusted Gross"),
    )
    .properties(title="Average Adjusted Gross by Disney Director")
)
director_gross_plot


# Figure 2. Bar chart of Disney Directors Average Adjusted gross of their Disney Movies

# ## Discussions

# I analyzed the Disney movie datasets and tried to determine which disney directors movies had the highest average gross. I looked at the adjusted gross after inflation since many of the disney movies where released in the mid 1900s and the adjusted gross would be more relevant than looking at the total gross. First I checked which genre of disney movie has the highest average gross after inflation. I wanted to look at this because there are many factors which can influence a movies gross. Disney musicals where significantly higher grossing on average. This was not too suprising since there are many popular disney musicals. However, what was suprising was the magnitude of difference between Disney musicals and Disney adventures, the second highest average grossing genre. Disney musicals were on average more than three times higher grossing than Disney adventures. 
# 
# David Hand on had the highest average grossing of his disney movies. This was about Two times greater than the next director with the next highest average grossing disney films. It would be interesting to look further into David Hand's Disney movies and what makes his movies popular. 
# 
# With the directors average gross for their disney films you can further examine his individual films to see if they gross above or below average. This could be helpful for looking at other factor which can effect how much a film may gross. 

# ## References

# online resources used: 
# {cite}`5`
# {cite}`6`
# 

# In[ ]:





# In[ ]:




