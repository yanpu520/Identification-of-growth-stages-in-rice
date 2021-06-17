#!/usr/bin/env python
# coding: utf-8

# In[1]:


def corr_map(df):
    var_corr = df.corr()
    mask = np.zeros_like(var_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    f, ax = plt.subplots(figsize=(20, 12))
    sns.set(font_scale=1)
    sns.heatmap(var_corr, mask=mask, cmap=cmap, vmax=1, center=0
               ,square=True, linewidths=.5, cbar_kws={"shrink": .5}
               ,annot=True,annot_kws={'size':12,'weight':'bold', 'color':'red'})
    plt.savefig('res.png', dpi=300)
    plt.show()   


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\显著性检验后的期刊数据.xls',sheet_name=1)
data


# In[5]:


data=data.iloc[:,0:16]


# In[6]:


data


# In[7]:


corr_map(data)


# In[8]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[9]:


corr=data.corr()
corr


# In[10]:


plt.rcParams['font.sans-serif']=['SimHei']
fig = plt.figure(figsize=(10,10))
sns.heatmap(corr)
plt.show()


# In[1]:


plt.rcParams['font.sans-serif']=['SimHei']
fig = plt.figure(figsize=(10,10))
sns.heatmap(corr,annot = True)
plt.rcParams['axes.unicode_minus']=False#解决负号显示问题
plt.savefig('000.png')
plt.show()


# In[ ]:




