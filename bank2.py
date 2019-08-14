#!/usr/bin/env python
# coding: utf-8

# In[27]:


#顧客情報から融資できるか予測
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
import mglearn
# 実行上問題ない注意は非表示にする
warnings.filterwarnings('ignore') 


# In[2]:


#csvファイルの読み込み
df = pd.read_csv("data/bank_01.csv")


# In[3]:


#ワンホットエンコーディング
data_dummies = pd.get_dummies(df)
data_dummies.columns
data_dummies.head()


# In[4]:


features = data_dummies.loc[:, 'age':'poutcome_success']
X = features.values
y = data_dummies['y_no'].values
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
print(data_dummies.y_no.value_counts())


# In[32]:


pipe = Pipeline([('scaler', StandardScaler()), ('classifier', SVC())])
param_grid = [
    {'classifier': [SVC()], 'classifier__C': [0.001, 0.01, 0.1, 1]},
    {'classifier': [RandomForestClassifier()], 'scaler': [None], 'classifier__max_features': [1, 2, 3, 4, 5]}
]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))


# In[33]:


#確認用
#grid.cv_results_ 


# In[51]:


pipe = Pipeline([('scaler', StandardScaler()), ('classifier', SVC())])
param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1], 'classifier__gamma': [0.01, 0.1, 1, 10, 100]}    

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, return_train_score=False, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set AUC: {:.2f}".format(grid.score(X_test, y_test)))


# In[52]:


xa = 'classifier__gamma'
xx = param_grid[xa]
ya = 'classifier__C'
yy = param_grid[ya]
plt.figure(figsize=(5,8))
scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(yy), -1)
mglearn.tools.heatmap(scores, xlabel=xa, xticklabels=xx, 
                      ylabel=ya, yticklabels=yy, cmap="viridis")


# In[ ]:


#確認用
"""
svc1 = SVC(C=0.01, gamma=1)
svc1.fit(X_train , y_train)
svc1.score(X_test, y_test)
roc_auc_score(y_test, svc1.predict(X_test))
"""


# In[42]:


#確認用
#grid.cv_results_ 

