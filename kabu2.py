#!/usr/bin/env python
# coding: utf-8

# In[1]:


#上場企業業種バラバラの７０社の過去２０年分データから翌日の終値の上昇率を予測
#高スコアとなるよう得微量を増やす
#１．前日の終値に比べた始値、高値、安値の得微量追加
# 1_1. 予測結果の合計を計算（空売り込）
# 1_2 予測結果の合計を計算（空売り無し。買いだけ）
#２．それぞれの得微量の割合を追加
#３．転換線、基準線、先行スパン1、先行スパン2を追加
#４　i日分の前からの上昇率を５日分追加
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import *
import seaborn as sns
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import random
import warnings
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


#業種バラバラの上場企業70社の過去２０年分株価データ
co_list = [1377, 1380, 1758, 1771, 1773, 1775, #1776, 1841, 1847, 1852, 1853, 1873, 1878, 1881, 
           2058, 2201, 2224, 2268, 2288, 2291, #2501, 2502, 2521, 2579, 2593, 2594, 2601, 2612, 
           3116, 3121, 3125, 3205, 3302, 3420, #3421, 3524, 3526, 3571, 2593, 3861, 3891, 3948,
           4026, 4078, 4088, 4113, 4118, 4151, #4187, 4229, 4274, 4366, 4403, 4406, 4516, 4547, 
           5008, 5010, 5101, 5122, 5162, 5186, #5191, 5201, 5202, 5212, 5273, 5304, 5355, 5357,
           6874, 6875, 6877, 6889, 6890, 6897, #6941, 6942, 6943, 6944, 6945, 6946, 6947, 6982,
           7860, 7862, 7865, 7873, 7874, 7893, #7894, 7895, 7899, 7908, 7918, 7921, 7932, 7950, 
           8001, 8002, 8028, 8039, 8045, 8053, #8072, 8072, 8078, 8095, 8096, 8153, 8160, 8171,
           9719, 9720, 9728, 9729, 9733, 9733, #9734, 9746, 9749, 9753, 9755, 9757, 9758, 9759, 
           9760, 9776, 9795, 9830, 9830, 9831, #9832, 9832, 9835, 9887, 9889, 9928, 9930, 9928, 
           9930, 9932, 9960, 9962, 9969, 9972, #9973, 9974, 9976, 9977, 9978, 9979, 9980, 9982, 
           9983, 9984, 9986, 9987#, 9989
          ]


# In[3]:


#０。基準となる得微量
#base日以降のデータを使う
base = 100
day_ago = 25
num_sihyou = 8 
reset =True
for co in co_list:
    temp = pd.read_csv("data/kabu1/" + str(co) + ".csv", header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp2), 0] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 1] #始値の前日からの上昇率
    temp3[0:len(temp2), 1] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 2] #高値の前日からの上昇率
    temp3[0:len(temp2), 2] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 3] #安値の前日からの上昇率
    temp3[0:len(temp2), 3] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4] #終値の前日からの上昇率
    temp3[0:len(temp2), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base-1:len(temp2)-1, 5].astype(np.float) #５日平均の前日からの上昇率
    temp3[0:len(temp2), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base-1:len(temp2)-1, 6].astype(np.float) #２５日平均の前日からの上昇率
    temp3[0:len(temp2), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base-1:len(temp2)-1, 7].astype(np.float) #７５日平均の前日からの上昇率
    temp3[0:len(temp2), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float) #出来高の前日からの上昇率
    
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
            
    # 説明変数となる行列Xを作成
    if reset:
        X = tempX
        reset = False
    else:
        X = np.concatenate((X, tempX), axis=0)
X = X[base:]

# 被説明変数となる Y = pre_day後の終値-当日終値 を作成
y = np.zeros(len(X))
# 何日後を値段の差を予測する
pre_day = 1
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]


# In[28]:


pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', linear_model.LinearRegression())])
param_grid = [
    {'scaler': [MinMaxScaler(), None]}
]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3 ,return_train_score=False)
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# In[3]:


#１．前日の終値に比べた始値、高値、安値の得微量追加
base = 100
day_ago = 25
num_sihyou = 11
reset =True
for co in co_list:
    temp = pd.read_csv("data/kabu1/" + str(co) + ".csv", header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp2), 0] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 1] #始値の前日からの上昇率
    temp3[0:len(temp2), 1] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 2] #高値の前日からの上昇率
    temp3[0:len(temp2), 2] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 3] #安値の前日からの上昇率
    temp3[0:len(temp2), 3] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4] #終値の前日からの上昇率
    temp3[0:len(temp2), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base-1:len(temp2)-1, 5].astype(np.float) #５日平均の前日からの上昇率
    temp3[0:len(temp2), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base-1:len(temp2)-1, 6].astype(np.float) #２５日平均の前日からの上昇率
    temp3[0:len(temp2), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base-1:len(temp2)-1, 7].astype(np.float) #７５日平均の前日からの上昇率
    temp3[0:len(temp2), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float) #出来高の前日からの上昇率
    
    temp3[0:len(temp2), 8] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた始値
    temp3[0:len(temp2), 9] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた高値
    temp3[0:len(temp2), 10] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた安値
    
    
    # 説明変数となる行列Xを作成します
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
    if reset:
        X = tempX
        reset = False
    else:
        X = np.concatenate((X, tempX), axis=0)
X = X[base:]

# 被説明変数となる Y = pre_day後の終値-当日終値 を作成
y = np.zeros(len(X))
# 何日後を値段の差を予測するのか
pre_day = 1
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]


# In[4]:


pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', linear_model.LinearRegression())])
param_grid = {'scaler': [MinMaxScaler(), None]}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3 ,return_train_score=False)
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# In[21]:


# 1_1. 予測結果の合計を計算（空売り込）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# 低い場合は終値で売り、翌日の終値で買う（空売り）：掛け金/翌日の上昇値
# 100日間
y_pred = grid.predict(X_test)
for j in range(0, 5):
    a=random.randrange(len(y_test)-100)
    X_test2 = X_test[a:a+100]
    y_test2 = y_test[a:a+100]
    y_pred2 = y_pred[a:a+100]
    money = 10000

    # 予測結果の総和グラフを描くーーーーーーーーー
    total_return = np.zeros(len(y_test2))
    for i in range(0,len(y_test2)): # len()で要素数を取得しています
        if y_pred2[i] > 1:
            money = money*y_test2[i]
        else:
            money = money/y_test2[i]
        total_return[i] = money
    

    print("投資結果：10000 円 → %1.3lf" %money, "円", "(正答率：%1.3lf" %grid.score(X_test2, y_test2), ")") 

plt.plot(total_return)


# In[22]:


# 1_2 予測結果の合計を計算（空売り無し。買いだけ）
# 上がると予測したら終値で買い,翌日の終値で売ったと想定：掛け金☓翌日の上昇値
# 100日間
for j in range(0, 5):
    a=random.randrange(len(y_test)-100)
    X_test2 = X_test[a:a+100]
    y_test2 = y_test[a:a+100]
    y_pred2 = y_pred[a:a+100]
    money = 10000

    # 予測結果の総和グラフを描くーーーーーーーーー
    total_return = np.zeros(len(y_test2))
    for i in range(0,len(y_test2)): # len()で要素数を取得しています
        if y_pred2[i] > 1:
            money = money*y_test2[i]
            
        total_return[i] = money

    print("投資結果：10000 円 → %1.3lf" %money, "円", "(正答率：%1.3lf" %grid.score(X_test2, y_test2), ")") 

plt.plot(total_return)


# In[4]:


#２．それぞれの得微量の割合を追加
base =100
day_ago = 25
num_sihyou = 20
reset =True
for co in co_list:
    temp = pd.read_csv("data/kabu1/" + str(co) + ".csv", header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp2), 0] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 1] #始値の前日からの上昇率
    temp3[0:len(temp2), 1] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 2] #高値の前日からの上昇率
    temp3[0:len(temp2), 2] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 3] #安値の前日からの上昇率
    temp3[0:len(temp2), 3] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4] #終値の前日からの上昇率
    temp3[0:len(temp2), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base-1:len(temp2)-1, 5].astype(np.float) #５日平均の前日からの上昇率
    temp3[0:len(temp2), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base-1:len(temp2)-1, 6].astype(np.float) #２５日平均の前日からの上昇率
    temp3[0:len(temp2), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base-1:len(temp2)-1, 7].astype(np.float) #７５日平均の前日からの上昇率
    temp3[0:len(temp2), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float) #出来高の前日からの上昇率
    
    temp3[0:len(temp2), 8] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた始値
    temp3[0:len(temp2), 9] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた高値
    temp3[0:len(temp2), 10] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた安値
    
    temp3[0:len(temp2), 11] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 4] #終値と始値の割合
    temp3[0:len(temp2), 12] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 4] #終値と高値の割合　
    temp3[0:len(temp2), 13] = temp2[base:len(temp2), 3] / temp2[base:len(temp2), 4] #終値と安値の割合　
    temp3[0:len(temp2), 14] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 3] #高値と始値の割合　
    temp3[0:len(temp2), 15] = temp2[base:len(temp2), 2] / temp2[base:len(temp2), 3] #高値と安値の割合　
    temp3[0:len(temp2), 16] = temp2[base:len(temp2), 1] / temp2[base:len(temp2), 2] #安値と始値の割合　
    
    temp3[0:len(temp2), 17] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 6].astype(np.float) #５日平均と２５日線の割合
    temp3[0:len(temp2), 18] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base:len(temp2), 7].astype(np.float) #５日平均と７５日線の割合
    temp3[0:len(temp2), 19] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base:len(temp2), 7].astype(np.float) #２５日平均と７５日線の割合
    
    # 説明変数となる行列Xを作成します
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
    if reset:
        X = tempX
        reset = False
    else:
        X = np.concatenate((X, tempX), axis=0)
X = X[base:]

# 被説明変数となる Y = pre_day後の終値-当日終値 を作成
y = np.zeros(len(X))
# 何日後を値段の差を予測するのか
pre_day = 1
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]


# In[5]:


pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', linear_model.LinearRegression())])
param_grid = {'scaler': [MinMaxScaler(), None]}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3 ,return_train_score=False)
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# In[8]:


#３．転換線、基準線、先行スパン1、先行スパン2を追加
base = 100
day_ago = 25
num_sihyou = 15
reset =True
for co in co_list:
    temp = pd.read_csv("data/kabu1/" + str(co) + ".csv", header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    temp3[0:len(temp2), 0] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 1] #始値の前日からの上昇率
    temp3[0:len(temp2), 1] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 2] #高値の前日からの上昇率
    temp3[0:len(temp2), 2] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 3] #安値の前日からの上昇率
    temp3[0:len(temp2), 3] = temp2[base:len(temp2), 4] / temp2[base-1:len(temp2)-1, 4] #終値の前日からの上昇率
    temp3[0:len(temp2), 4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base-1:len(temp2)-1, 5].astype(np.float) #５日平均の前日からの上昇率
    temp3[0:len(temp2), 5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base-1:len(temp2)-1, 6].astype(np.float) #２５日平均の前日からの上昇率
    temp3[0:len(temp2), 6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base-1:len(temp2)-1, 7].astype(np.float) #７５日平均の前日からの上昇率
    temp3[0:len(temp2), 7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-1:len(temp2)-1, 8].astype(np.float) #出来高の前日からの上昇率
    
    temp3[0:len(temp2), 8] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた始値
    temp3[0:len(temp2), 9] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた高値
    temp3[0:len(temp2), 10] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた安値
    
    # 一目均衡表を追加 (9,26, 52) 
    para1 =9
    para2 = 26
    para3 = 52
    # 転換線 = （過去(para1)日間の高値 + 安値） ÷ 2
    temp2a = np.c_[temp2, np.zeros((len(temp2),4))] # 列の追加
    for i in range(para1, len(temp2)):
        tmp_high = temp2[i-para1:i,2].astype(np.float)
        tmp_low = temp2[i-para1:i,3].astype(np.float)
        temp2a[i, 9] = (np.max(tmp_high) + np.min(tmp_low))  /2
    temp3[0:len(temp2), 11] = temp2a[base:len(temp2), 9] / temp2a[base-1:len(temp2)-1, 9]
        
    # 基準線 = （過去(para2)日間の高値 + 安値） ÷ 2
    for i in range(para2, len(temp2)):
        tmp_high = temp2[i-para2:i,2].astype(np.float)
        tmp_low = temp2[i-para2:i,3].astype(np.float)
        temp2a[i, 10] = (np.max(tmp_high) + np.min(tmp_low)) / 2 
    temp3[0:len(temp2), 12] = temp2a[base:len(temp2), 10] / temp2a[base-1:len(temp2)-1, 10]
        
    # 先行スパン1 = ｛ （転換値+基準値） ÷ 2 ｝を(para2)日先にずらしたもの
    for i in range(0, len(temp2)-para2):
        tmp =(temp2a[i,9] + temp2a[i,10]) / 2
        temp2a[i+para2, 11] = tmp
    temp3[0:len(temp2), 13] = temp2a[base:len(temp2), 11] / temp2a[base-1:len(temp2)-1, 11]
        
    # 先行スパン2 = ｛ （過去(para3)日間の高値+安値） ÷ 2 ｝を(para2)日先にずらしたもの
    for i in range(para3, len(temp2)-para2):
        tmp_high = temp2[i-para3+1:i+1, 2].astype(np.float)
        tmp_low = temp2[i-para3+1:i+1, 3].astype(np.float)
        temp2a[i+para2, 12] = (np.max(tmp_high) + np.min(tmp_low)) / 2 
    temp3[0:len(temp2), 14] = temp2a[base:len(temp2), 12] / temp2a[base-1:len(temp2)-1, 12]
    
    # 説明変数となる行列Xを作成
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
    if reset:
        X = tempX
        reset = False
    else:
        X = np.concatenate((X, tempX), axis=0)
X = X[base:]

# 被説明変数となる Y = pre_day後の終値-当日終値 を作成
y = np.zeros(len(X))
# 何日後を値段の差を予測するのか
pre_day = 1
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]


# In[9]:


pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', linear_model.LinearRegression())])
param_grid = {'scaler': [MinMaxScaler(), None]}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3 ,return_train_score=False)
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# In[3]:


#４　i日分の前からの上昇率を５日分追加
#base日以降のデータを使う
base = 100
day_ago = 25
n = 5
num_sihyou = 8* n + 3
reset =True
for co in co_list:
    temp = pd.read_csv("data/kabu1/" + str(co) + ".csv", header=0, encoding='cp932')
    temp = temp[['日付','始値', '高値','安値','終値','5日平均','25日平均','75日平均','出来高']]
    temp= temp.iloc[::-1]#上下反対
    temp2 = np.array(temp)
    temp3 = np.zeros((len(temp2)-base, num_sihyou))
    for i in range(0, n):
        temp3[0:len(temp2), i*8] = temp2[base:len(temp2), 1] / temp2[base-i-1:len(temp2)-i-1, 1] #始値のi日前からの上昇率
        temp3[0:len(temp2), i*8+1] = temp2[base:len(temp2), 2] / temp2[base-i-1:len(temp2)-i-1, 2] #高値のi日前からの上昇率
        temp3[0:len(temp2), i*8+2] = temp2[base:len(temp2), 3] / temp2[base-i-1:len(temp2)-i-1, 3] #安値のi日前からの上昇率
        temp3[0:len(temp2), i*8+3] = temp2[base:len(temp2), 4] / temp2[base-i-1:len(temp2)-i-1, 4] #終値のi日前からの上昇率
        temp3[0:len(temp2), i*8+4] = temp2[base:len(temp2), 5].astype(np.float) / temp2[base-i-1:len(temp2)-i-1, 5].astype(np.float) #５日平均のi日前からの上昇率
        temp3[0:len(temp2), i*8+5] = temp2[base:len(temp2), 6].astype(np.float) / temp2[base-i-1:len(temp2)-i-1, 6].astype(np.float) #２５日平均のi日前からの上昇率
        temp3[0:len(temp2), i*8+6] = temp2[base:len(temp2), 7].astype(np.float) / temp2[base-i-1:len(temp2)-i-1, 7].astype(np.float) #７５日平均のi日前からの上昇率
        temp3[0:len(temp2), i*8+7] = temp2[base:len(temp2), 8].astype(np.float) / temp2[base-i-1:len(temp2)-i-1, 8].astype(np.float) #出来高のi日前からの上昇率
    
    temp3[0:len(temp2), 40] = temp2[base:len(temp2), 1] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた始値
    temp3[0:len(temp2), 41] = temp2[base:len(temp2), 2] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた高値
    temp3[0:len(temp2), 42] = temp2[base:len(temp2), 3] / temp2[base-1:len(temp2)-1, 4] #前日の終値に比べた安値
    tempX = np.zeros((len(temp3), day_ago*num_sihyou))
    for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
        for i in range(0, day_ago):
            tempX[i:len(temp3), day_ago*s+i] = temp3[0:len(temp3)-i,s]
    # 説明変数となる行列Xを作成
    if reset:
        X = tempX
        reset = False
    else:
        X = np.concatenate((X, tempX), axis=0)
X = X[base:]

# 被説明変数となる Y = pre_day後の終値-当日終値 を作成
y = np.zeros(len(X))
# 何日後を値段の差を予測
pre_day = 1
y[0:len(y)-pre_day] = X[pre_day:len(X),0]
X = X[:-pre_day]
y = y[:-pre_day]


# In[4]:


pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', linear_model.LinearRegression())])
param_grid = {'scaler': [MinMaxScaler(), None]}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3 ,return_train_score=False)
grid.fit(X_train, y_train)
print("grid best score, ", grid.best_score_)
print("Best parameters: ", grid.best_params_)
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))


# In[ ]:





# In[ ]:





# In[ ]:




