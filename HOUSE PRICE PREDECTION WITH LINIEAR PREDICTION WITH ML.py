#!/usr/bin/env python
# coding: utf-8

# # HOUSE PRICE PREDECTION WITH LINIEAR PREDICTION WITH ML

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model


# In[2]:


df=pd.read_csv("homeprice.csv")
df


# In[ ]:





# In[3]:


#Distribute my DataFrame in PLOT


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


plt.scatter(df.area,df.price,color='g',marker="o")
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price (in dollar)")
plt.title("House Price Dataframe")


# In[14]:


reg = linear_model.LinearRegression()


# In[15]:


reg.fit(df[['area']],df.price)


# In[19]:


reg.predict([[3300]])


# In[20]:


reg.coef_


# In[22]:


reg.intercept_


# In[24]:


y=3300*(reg.coef_)+reg.intercept_


# In[25]:


y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


d=pd.read_csv("area.csv")


# In[48]:


d.head()


# In[49]:


p=reg.predict(d)


# In[50]:


d['Predict price ']=p


# In[52]:


d


# In[54]:


d.to_csv("Predicted price in that given area")


# In[55]:


e=pd.read_csv("Predicted price in that given area")


# In[56]:


e


# In[ ]:




