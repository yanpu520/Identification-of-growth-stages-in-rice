#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_excel(r'C:\Users\Administrator\Desktop\SFD.xls',sheet_name=1)
data


# In[3]:


data = data.dropna(how='any').copy()#去除含有缺失值的行
data


# In[4]:


u = data.mean()  # 计算均值
std = data.std()  # 计算标准差
for i in range(7):
    data =data[np.abs(data.iloc[:,i] - u[i]) <= 3*std[i]]
print(data)
#根据3σ准则来去除异常值


# In[5]:


train_X=data.iloc[:,0:-1]
train_Y=data.iloc[:,-1]
train_X=(train_X-train_X.mean())/train_X.std()#标准化处理


# In[6]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = SVC()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[7]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = GaussianNB()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[8]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = LDA()
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[9]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
accuracy


# In[10]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
lr = AdaBoostClassifier(n_estimators=50, base_estimator=None,learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[11]:


X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)


# In[12]:


X_train=np.array(X_train)
X_valid=np.array(X_valid)
Y_train=np.array(Y_train)
Y_valid=np.array(Y_valid)


# In[13]:


class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, Y_train, sample_weight):

        # 规范化样本权重（如果尚未规范化）
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        Y_train_resampled = np.zeros((len(Y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # 从0到len（X_train）-1画一个数字
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # 将绘制编号处的X和y放入重新采样的X和y中 
            X_train_resampled[i] = X_train[draw]
            Y_train_resampled[i] = Y_train[draw]

        return X_train_resampled, Y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


# In[16]:


lr = AdaBoostClassifier(n_estimators=150,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[19]:


lr = AdaBoostClassifier(n_estimators=50,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[20]:


lr = AdaBoostClassifier(n_estimators=50,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[21]:


lr = AdaBoostClassifier(n_estimators=50,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[11]:


lr = AdaBoostClassifier(n_estimators=100,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[12]:


lr = AdaBoostClassifier(n_estimators=100,base_estimator=customMLPClassifer(),learning_rate=1,algorithm='SAMME')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_valid)
accuracy = np.mean(Y_predict == Y_valid) * 100
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(lr, X_train, Y_train, cv=kfold)#对数据进行十折交叉验证--9份训练，一份测试
results


# In[26]:


train_Y = label_binarize(train_Y, classes=[0, 1, 2])
Y_valid =  label_binarize(Y_valid , classes=[0, 1, 2])
train_Y


# In[27]:


n_classes = train_Y.shape[1]
n_classes


# In[36]:


lr = AdaBoostClassifier(n_estimators=100, base_estimator=None,learning_rate=1,algorithm='SAMME')
y_score = lr.fit(X_train, Y_train).decision_function(X_valid)


# In[37]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[38]:


fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[39]:


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


# In[40]:


mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])


# In[41]:


mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# In[42]:


from itertools import cycle
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




