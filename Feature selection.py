#!/usr/bin/env python
# coding: utf-8

# In[4]:


from scipy import stats
import pandas as pd
import numpy as np


# In[16]:


df1=pd.read_excel(r'C:\Users\Administrator\Desktop\修正后的数据.xls',sheet_name=1)
df2=pd.read_excel(r'C:\Users\Administrator\Desktop\修正后的数据.xls',sheet_name=2)
df3=pd.read_excel(r'C:\Users\Administrator\Desktop\修正后的数据.xls',sheet_name=3)
df3


# In[29]:


columns=df1.columns.values.tolist() 
columns


# In[35]:


for i in range(1,34):
        print(columns[i]+"的显著性检验为：")
        print(stats.ttest_ind(df1.iloc[:,i],df2.iloc[:,i]))


# In[36]:


for i in range(1,34):
        print(columns[i]+"的显著性检验为：")
        print(stats.ttest_ind(df2.iloc[:,i],df3.iloc[:,i]))


# In[37]:


for i in range(1,34):
        print(columns[i]+"的显著性检验为：")
        print(stats.ttest_ind(df1.iloc[:,i],df3.iloc[:,i]))


# In[ ]:




