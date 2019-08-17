#!/usr/bin/env python
# coding: utf-8

# In[1]:


#顧客情報から融資できるか予測
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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


# In[5]:


pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(max_iter=200000))])
param_grid = {'classifier__alpha':[0.01, 0.1, 1, 10 ], 'classifier__hidden_layer_sizes': [[10], [100], [1000], [10000], [100000]]}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=2, return_train_score=False,  scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))


# In[6]:


#確認用
#results = pd.DataFrame(grid.cv_results_)
#results[:5]
grid.cv_results_ 


# In[8]:


xa = 'classifier__hidden_layer_sizes'
xx = param_grid[xa]
ya = 'classifier__alpha'
yy = param_grid[ya]
plt.figure(figsize=(5,8))
scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(yy), -1)
mglearn.tools.heatmap(scores, xlabel=xa, xticklabels=xx, 
                      ylabel=ya, yticklabels=yy, cmap="viridis")


# In[ ]:


pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(max_iter=200000))])
param_grid = {'classifier__alpha':[10, 50, 100, 1000 ], 'classifier__hidden_layer_sizes': [[10], [100]]}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=2, return_train_score=False,  scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))


# In[ ]:


xa = 'classifier__hidden_layer_sizes'
xx = param_grid[xa]
ya = 'classifier__alpha'
yy = param_grid[ya]
plt.figure(figsize=(5,8))
scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(yy), -1)
mglearn.tools.heatmap(scores, xlabel=xa, xticklabels=xx, 
                      ylabel=ya, yticklabels=yy, cmap="viridis")


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


# lbfg
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(solver='lbfgs', max_iter=200000))])
param_grid = {'classifier__hidden_layer_sizes': [[1000], [10000], [100000]], 'classifier__alpha': [0.01, 0.1, 1, 10]}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=2, return_train_score=False,  scoring="roc_auc")
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))


# In[ ]:


xa = 'classifier__hidden_layer_sizes'
xx = param_grid[xa]
ya = 'classifier__alpha'
yy = param_grid[ya]
plt.figure(figsize=(5,8))
scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(yy), -1)
mglearn.tools.heatmap(scores, xlabel=xa, xticklabels=xx, 
                      ylabel=ya, yticklabels=yy, cmap="viridis")


# In[ ]:




