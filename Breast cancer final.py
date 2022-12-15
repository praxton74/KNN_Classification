#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv(r'C:\Users\smara\AppData\Local\Microsoft\Windows\INetCache\IE\JMNQTN2W\archive[1].zip')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.columns


# In[7]:


data.shape


# In[10]:


y=data['diagnosis']
x=data[['id',  'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]


# In[11]:


x.shape


# In[12]:


y.shape


# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[14]:


print("Shape of X_train= ",x_train)


# In[15]:


print("Shape of Y_train= ",y_train)


# In[16]:


print("Shape of X_test= ",x_test)


# In[17]:


print("Shape of _test= ",y_test)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


model=KNeighborsClassifier(n_neighbors=5)


# In[26]:


model.fit(x_train,y_train)


# In[27]:


model.score(x_test,y_test)


# In[ ]:




