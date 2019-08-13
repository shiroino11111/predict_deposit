#!/usr/bin/env python
# coding: utf-8

# In[16]:


#顧客情報から融資できるか予測
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
import mglearn
# 実行上問題ない注意は非表示にする
warnings.filterwarnings('ignore') 


# In[17]:


#csvファイルの読み込み
df = pd.read_csv("data/bank_01.csv")


# In[18]:


#ワンホットエンコーディング
data_dummies = pd.get_dummies(df)
data_dummies.columns
data_dummies.head()


# In[19]:


features = data_dummies.loc[:, 'age':'poutcome_success']
X = features.values
y = data_dummies['y_no'].values
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
print(data_dummies.y_no.value_counts())


# In[20]:


pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])
param_grid = {'classifier__C':[0.001, 0.01, 0.1, 1, 10]}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, return_train_score=False,  scoring="roc_auc")
grid.fit(X_train, y_train)
print("Best parameters: ", grid.best_params_)
print("grid best score, ", grid.best_score_)
print("Test set AUC: {:.2f}".format(grid.score(X_test, y_test)))


# In[21]:


#確認用
grid.cv_results_ 


# In[ ]:




