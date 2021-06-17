#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\期刊数据.xls',sheet_name=1)
data


# In[3]:


data = data.dropna(how='any').copy()#去除含有缺失值的行
data


# In[4]:


u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
for i in range(12):
    data =data[np.abs(data.iloc[:,i] - u[i]) <= 3*std[i]]
print(data)
#根据3σ准则来去除异常值


# In[5]:


train_X=data.iloc[:,0:-1]
train_Y=data.iloc[:,-1]
train_X=(train_X-train_X.mean())/train_X.std()#标准化处理


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=123)


# In[7]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel


# In[8]:


data=data.drop(['类别'], axis=1)


# In[9]:


model = RandomForestRegressor(random_state=2, max_depth=15)
data=pd.get_dummies(data)
model.fit(data,train_Y)


# In[12]:


features = data.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:16]) # top 15 features
plt.title('Feature Importances')
plt.rcParams['font.sans-serif']='SimHei'#将字体设置为中文
plt.rcParams['axes.unicode_minus']=False#解决负号显示问题
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("cd.jpeg",dpi = 1000)
plt.show()


# In[ ]:




