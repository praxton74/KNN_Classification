#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('breast-cancer.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


del data['id']


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[8]:


plt.figure(figsize=(15,5))
sns.countplot(data['diagnosis'])


# In[9]:


data['diagnosis'].value_counts(normalize=True)


# In[10]:


x=data.iloc[ : ,3:25]
y=data['diagnosis']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


# In[12]:


a_score=[]
for k in range(1,18,2):
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train,y_train)
  y_pred=knn.predict(x_test)
  accuracy=accuracy_score(y_test,y_pred)
  a_score.append(accuracy)

mse=[1 - x for x in a_score]


# In[13]:


max=a_score[0]
for j in a_score:
  if(j>max):
    max=j
print(max)


# In[ ]:




