#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[34]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\SFD.xls',sheet_name=1)


# In[35]:


data=data.iloc[:,-2]


# In[36]:


data1=data.iloc[0:423]
data2=data.iloc[423:825]
data3=data.iloc[825:]


# In[37]:


s1 = pd.Series(np.array(data1))
s2 = pd.Series(np.array(data2))
s3 = pd.Series(np.array(data3))


# In[38]:


data = pd.DataFrame({"1": s1, "2": s2, "3": s3})
data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
plt.ylabel("The value of the SFD")
plt.xlabel("Growth stage")  # 我们设置横纵坐标的标题。
plt.savefig("SFD.png")
plt.show()


# In[39]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\FBC.xls',sheet_name=1)


# In[40]:


data=data.iloc[:,-2]
data1=data.iloc[0:423]
data2=data.iloc[423:825]
data3=data.iloc[825:]


# In[41]:


s1 = pd.Series(np.array(data1))
s2 = pd.Series(np.array(data2))
s3 = pd.Series(np.array(data3))


# In[42]:


data = pd.DataFrame({"1": s1, "2": s2, "3": s3})
data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
plt.ylabel("The value of the FBC")
plt.xlabel("Growth stage")  # 我们设置横纵坐标的标题。
plt.savefig("FBC.png")
plt.show()


# In[43]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\DBCS.xls',sheet_name=1)
data=data.iloc[:,-2]
data1=data.iloc[0:423]
data2=data.iloc[423:825]
data3=data.iloc[825:]
s1 = pd.Series(np.array(data1))
s2 = pd.Series(np.array(data2))
s3 = pd.Series(np.array(data3))
data = pd.DataFrame({"1": s1, "2": s2, "3": s3})
data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
plt.ylabel("The value of the DBCS")
plt.xlabel("Growth stage")  # 我们设置横纵坐标的标题。
plt.savefig("DBCS.png")
plt.show()


# In[44]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\DBCG.xls',sheet_name=1)
data=data.iloc[:,-2]
data1=data.iloc[0:423]
data2=data.iloc[423:825]
data3=data.iloc[825:]
s1 = pd.Series(np.array(data1))
s2 = pd.Series(np.array(data2))
s3 = pd.Series(np.array(data3))
data = pd.DataFrame({"1": s1, "2": s2, "3": s3})
data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
plt.ylabel("The value of the DBCG")
plt.xlabel("Growth stage")  # 我们设置横纵坐标的标题。
plt.savefig("DBCG.png")
plt.show()


# In[ ]:




