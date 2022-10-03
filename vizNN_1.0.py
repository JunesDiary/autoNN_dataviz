#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install hvplot')


# In[3]:


get_ipython().system('pip install plotly')


# In[2]:


#importing and setting up
import numpy as np
import hvplot.dask
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
import dask.dataframe as dd

import re

hvplot.extension('plotly')


# In[3]:


#loading onto dask dataframe
df = dd.read_csv('car_design.csv')
df_original = df

#specify target label
target_label = 4
df.head()  #check dataframe loaded


# In[4]:


def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


# create a generalisation for the data needed for the plotting, by changing the labels


#1 set the column length count
length = len(df.columns)
original_labels = df.columns
#2 rename labels of features to generalise them
generated_labels = ["Label#{0}".format(i) for i in range(0, length)]
df = df.rename(columns=dict(zip(df.columns, generated_labels)))
df.head() #check renamed column 




# In[5]:


#creating non-numeric column identifiers
non_numeric_col_pos = np.empty(length, dtype=object)
non_numeric_col_pos_counter = 0


#displays boxplot for all columns to identify outliers
#also performs segregation of numeric and non-numeric columns
for i in range(0, length):
    box_label = "Label#{0}".format(i) #i = column of interest postion (first column starting from 0)
    try:
        box_features = np.array(df[box_label]).astype('float64')
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(box_features)

        # Adding title and axes
        current_label = original_labels[i]
        plt.title("Box plot for "+ current_label)
        plt.xlabel('Plot')
        plt.ylabel(current_label)
        plt.show()
        
    except Exception as e:
        if findWholeWord('could not convert string to float')(str(e)):
            non_numeric_col_pos[non_numeric_col_pos_counter] = i   #array containing non numeric column positions
            non_numeric_col_pos_counter = non_numeric_col_pos_counter + 1   
            
            
#BOXPLOTTING & NON NUMERIC COLUMN IDENTIFYING SUCCESS ~~ ARJUN


# In[57]:


#non-numeric column categorical plotting

while non_numeric_col_pos[non_numeric_col_pos_counter] != None:
    


# In[6]:


#plots the correlation matrix for the data 
heatmap = np.empty((length - non_numeric_col_pos_counter), dtype=object)
#separate numeric column labels for heatmap plot axes labeling
j = np.arange(length)
numeric_counter = 0
for i in range(0, length):
    if j[i] in non_numeric_col_pos:
        None
    else:
        heatmap[numeric_counter] = original_labels[i]
        numeric_counter = numeric_counter + 1
ax = sns.heatmap(df.corr(), annot=True, xticklabels=list(heatmap), yticklabels=list(heatmap), annot_kws={"size": 35 / np.sqrt(len(df.corr()))})


# In[9]:


#plots the Plot pairwise relationships in a dataset

    
sns.pairplot(df_original.compute(), hue="price", diag_kind="hist",height = 3)


# In[ ]:




