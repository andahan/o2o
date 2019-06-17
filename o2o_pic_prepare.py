#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'preprocess_0610'

__author__ = 'anhan'

import os, sys, pickle
import math
import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from datetime import date

# from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
# from sklearn.preprocessing import MinMaxScaler

# import xgboost as xgb
# import lightgbm as lgb



def date_gap(row):
    if row['Date'] != -1 and row['Date_received'] != -1:
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        # if td <= pd.Timedelta(15, 'D'):
        # print(td.days)
        return td.days
    else:
        return -1


'''
距离分布图
'''
def get_pic_distance(dfoff):
    countbydistance= dfoff[dfoff['Distance'] != -1].groupby('Distance').size().reset_index(name='count')
# print(countbydistance)
#     Distance   count
# 0        0.0  826070
# 1        1.0  227221
# 2        2.0  118413
# 3        3.0   76598
# 4        4.0   55085
# 5        5.0   41452
# 6        6.0   32483
# 7        7.0   25681
# 8        8.0   21436
# 9        9.0   17958
# 10      10.0  206484
    sns.set_style('ticks')
    sns.set_context("notebook", font_scale= 1.4)
    plt.figure(figsize = (12,8))

    plt.bar(countbydistance['Distance'], countbydistance['count']) #(x,y,label)
# # plt.yscale('log') #y用对数展示
    plt.xlabel('Location Resistance (Distance/500)')
    plt.ylabel('Count')

    plt.savefig('./pic/pic_distance.jpg')




'''
券时间序列分布图
'''
def get_pic_coupon_day(dfoff):
    date_received = dfoff['Date_received'].unique()
    date_received = sorted(date_received[date_received != -1])

    couponbydate = dfoff[dfoff['Date_received'] != -1][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    couponbydate.columns = ['Date_received','count'] #按照券收到时间汇总,每天券被领取的总量

    buybydate = dfoff[(dfoff['Date'] != -1) & (dfoff['Date_received'] != -1)][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received','count'] #按照券收到时间汇总,每天券被领取且该券后来被消费的数量


    sns.set_style('ticks')
    sns.set_context("notebook", font_scale= 1.4)
    plt.figure(figsize = (12,8))
    date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

    # 激活第1个 subplot
    plt.subplot(211) #subplot(numRows, numCols, plotNum) 图表的整个绘图区域被分成 numRows 行和 numCols 列，此处表示一共是2行1列的图，此
    plt.plot(date_received_dt, couponbydate['count'], linestyle='-', color='b',label = 'number of coupon received') #(x,y,label)
    plt.plot(date_received_dt, buybydate['count'], linestyle='-.', color='r',label = 'number of coupon used')

    plt.yscale('log') #y用对数展示
    plt.xlabel('Date_received')
    plt.ylabel('Count')
    plt.legend() #显示图例（此处为两个label的说明）

    # 激活第2个 subplot
    plt.subplot(212)
    plt.plot(date_received_dt, buybydate['count']/couponbydate['count'])
    plt.xlabel('Date_received')
    plt.ylabel('Ratio(coupon used/coupon received)')
    plt.tight_layout() #tight_layout()调整子图之间的间隔来减少重复叠放。

    plt.savefig('./pic/pic_coupon_day.jpg')




'''
date gap分布图
'''
def get_pic_date_gap(dfoff):
    countbydistance= dfoff[dfoff['date_gap'] != -1].groupby('date_gap').size().reset_index(name='count')
    sns.set_style('ticks')
    sns.set_context("notebook", font_scale= 1.4)
    plt.figure(figsize = (12,8))

    plt.bar(countbydistance['date_gap'], countbydistance['count']) #(x,y,label)
    plt.yscale('log') #y用对数展示
    plt.xlabel('Coupon Redemption Duration')
    plt.ylabel('Count')

    plt.savefig('./pic/pic_date_gap.jpg')



'''
调节效应图
'''
def get_pic_moderate():

    sns.set_style('ticks')
    sns.set_context("notebook", font_scale= 1.4)
    plt.figure(figsize = (12,6))

    #位置阻力
    dx = np.arange(1, 11)
    dyhigh = -0.12 * dx + 9.23
    dylow = -0.05 * dx + 8.4
    #搜索阻力
    sx = np.arange(1, 11)
    syhigh = -0.07 * sx + 8.31
    sylow = -0.16 * sx + 8.74


    plt.subplot(121)
    plt.xticks((), ())#设置坐标轴上不显示坐标值标签
    plt.yticks((), ())
    plt.yticks((), ())

    plt.plot(dx, dyhigh,linestyle='-', color='b',label = 'Location Resistance (High)')
    plt.plot(dx, dylow, linestyle='-.', color='r', label = 'Location Resistance (Low)')
    plt.ylabel('Coupon Redemption Duration')
    plt.xlabel('Connection')
    plt.legend() #显示图例（此处为两个label的说明）


    plt.subplot(122)
    plt.xticks((), ())
    plt.yticks((), ())
    plt.yticks((), ())

    plt.plot(sx, syhigh,linestyle='-', color='b',label='Search Resistance (High)')
    plt.plot(sx, sylow,linestyle='-.', color='r',label='Search Resistance (Low)')
    plt.ylabel('Coupon Redemption Duration')
    plt.xlabel('Connection')
    plt.legend() #显示图例（此处为两个label的说明）

    plt.savefig('./pic/pic_moderate.jpg')








if __name__ == '__main__':
    dfoff = pd.read_csv('./o2o_data/ccf_offline_stage1_train.csv')
    # nan --> -1
    dfoff = dfoff.fillna(-1)

    # temporal process
    dfoff['date_gap'] = dfoff.apply(date_gap, axis=1)

    # get_pic_coupon_day(dfoff)
    # get_pic_distance(dfoff)
    # get_pic_date_gap(dfoff)

    get_pic_moderate()


    # print(len(dfoff['User_id'].unique()))
    # # [output]:539438
    #
    # print(len(dfoff['Merchant_id'].unique()))
    # # [output]:8415



