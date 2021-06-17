#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,RidgeCV,Lasso, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import  metrics as mt
from sklearn import svm
from sklearn.svm import SVC


# In[2]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\最终数据.xls',sheet_name=1)
data


# In[3]:


data = data.dropna(how='any').copy()#去除含有缺失值的行
data


# In[4]:


u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
for i in range(16):
    data =data[np.abs(data.iloc[:,i] - u[i]) <= 3*std[i]]
print(data)
#根据3σ准则来去除异常值


# In[5]:


data


# In[6]:


train_X=data.iloc[:,0:-1]
train_Y=data.iloc[:,-1]
train_X=(train_X-train_X.mean())/train_X.std()#标准化处理


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[42]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=123,stratify=train_Y)#不平衡策略
lr = SVC()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[43]:


mt.f1_score(Y_valid, Y_predict,average='macro')


# In[13]:


Y_train


# In[14]:


data.corr()


# In[18]:


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


# In[31]:


model = RandomForestRegressor(random_state=1, max_depth=10)
data=pd.get_dummies(train_X)
model.fit(train_X,train_Y)


# In[41]:


features = train_X.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:16]) # top 15 features
plt.title('Feature Importances')
plt.rcParams['font.sans-serif']='SimHei'#将字体设置为中文
plt.rcParams['axes.unicode_minus']=False#解决负号显示问题
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:




