#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import  metrics as mt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold


# In[12]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\期刊全部数据.xls',sheet_name=1)
data


# In[13]:


data = data.dropna(how='any').copy()#去除含有缺失值的行
data


# In[14]:


u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
for i in range(7):
    data =data[np.abs(data.iloc[:,i] - u[i]) <= 3*std[i]]
print(data)
#根据3σ准则来去除异常值


# In[15]:


train_X=data.iloc[:,0:-1]
train_Y=data.iloc[:,-1]
train_X=(train_X-train_X.mean())/train_X.std()#标准化处理


# In[21]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = SVC()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[22]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = GaussianNB()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[27]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = LDA()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[28]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[30]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = AdaBoostClassifier(n_estimators=50, base_estimator=None,learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[ ]:




