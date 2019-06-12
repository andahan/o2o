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

# import seaborn as sns

from datetime import date

# from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
# from sklearn.preprocessing import MinMaxScaler

# import xgboost as xgb
# import lightgbm as lgb


'''
dfoff: 175 4884, User_id,Merchant_id,Coupon_id,Discount_rate,Distance,Date_received,Date
dfon: 1142 9826, User_id,Merchant_id,Action,Coupon_id,Discount_rate,Date_received,Date
'''


# print(dfoff.info())
# User_id          int64
# Merchant_id      int64
# Coupon_id        float64
# Discount_rate    object
# Distance         float64
# Date_received    float64
# Date             float64

# print(dfon.info())
# User_id          int64
# Merchant_id      int64
# Action           int64
# Coupon_id        object
# Discount_rate    object
# Date_received    float64
# Date             float64

# off优惠券使用情况：
# print('有优惠券，购买商品条数', dfoff[(dfoff['Date_received'] != -1) & (dfoff['Date'] != -1)].shape[0])
# print('无优惠券，购买商品条数', dfoff[(dfoff['Date_received'] == -1) & (dfoff['Date'] != -1)].shape[0])
# print('有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] != -1) & (dfoff['Date'] == -1)].shape[0])
# print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == -1) & (dfoff['Date'] == -1)].shape[0])

# [output]:
# 有优惠券，购买商品条数 7 5382
# 无优惠券，购买商品条数 70 1602
# 有优惠券，不购买商品条数 97 7900
# 无优惠券，不购买商品条数 0


# print('Discount_rate 种类:',dfoff['Discount_rate'].unique())
# print('Distance 种类:', dfoff['Distance'].unique())

# [output]:
# Discount_rate 种类: [-1 '150:20' '20:1' '200:20' '30:5' '50:10' '10:5' '100:10' '200:30'
#  '20:5' '30:10' '50:5' '150:10' '100:30' '200:50' '100:50' '300:30'
#  '50:20' '0.9' '10:1' '30:1' '0.95' '100:5' '5:1' '100:20' '0.8' '50:1'
#  '200:10' '300:20' '100:1' '150:30' '300:50' '20:10' '0.85' '0.6' '150:50'
#  '0.75' '0.5' '200:5' '0.7' '30:20' '300:10' '0.2' '50:30' '200:100'
#  '150:5']
# Distance 种类: [ 0.  1. -1.  2. 10.  4.  7.  9.  3.  5.  6.  8.]


'''
[Distance]
[时空关-空]
'''

'''
[Discount_rate]
处理方式：将str变成 numeric
将满xx减yy类型(xx:yy)的券变成折扣率 : 1 - yy/xx，
同时建立折扣券相关的特征 discount_rate, discount_man, discount_jian, discount_type
'''

def get_discount_type(row):
    if row == '-1':
        return -1
    elif ':' in row:
        return 1
    else:
        return 0  #0为打折，1为满减，-1为null

def convert_rate(row):
    """Convert discount to rate"""
    if row == '-1':
        return 1.0
    elif row == 'fixed':
        return 2.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def get_discount_man(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def get_discount_jian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


def process_discount_data(df):
    # convert discunt_rate
    df['o_discount_ratio'] = df['Discount_rate'].astype(str).apply(convert_rate)
    df['o_discount_man'] = df['Discount_rate'].astype(str).apply(get_discount_man)
    df['o_discount_jian'] = df['Discount_rate'].astype(str).apply(get_discount_jian)
    df['o_discount_type'] = df['Discount_rate'].astype(str).apply(get_discount_type)
    # print(df['discount_rate'].astype(str).unique())
    return df




# print(dfoff.head(10))
# [output]:

'''
[时空关-时]
两个时间，Date_received和Date，即优惠券收到日期和消费日期。
'''
# date_received = dfoff['Date_received'].unique()
# date_received = sorted(date_received[date_received != -1])
#
# date_buy = dfoff['Date'].unique()
# date_buy = sorted(date_buy[date_buy != -1])
#
# date_buy = sorted(dfoff[dfoff['Date'] != -1]['Date'])
# print('优惠券收到日期从',date_received[0],'到', date_received[-1])
# print('消费日期从', date_buy[0], '到', date_buy[-1])

# [output]:
# 优惠券收到日期从 20160101.0 到 20160615.0
# 消费日期从 20160101.0 到 20160630.0


# couponbydate = dfoff[dfoff['Date_received'] != -1][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
# couponbydate.columns = ['Date_received','count'] #按照券收到时间汇总,每天券被领取的总量
#
# buybydate = dfoff[(dfoff['Date'] != -1) & (dfoff['Date_received'] != -1)][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
# buybydate.columns = ['Date_received','count'] #按照券收到时间汇总,每天券被领取且该券后来被消费的数量
#
#
# sns.set_style('ticks')
# sns.set_context("notebook", font_scale= 1.4)
# plt.figure(figsize = (12,8))
# date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')
#
# # 激活第1个 subplot，画柱状图
# plt.subplot(211) #subplot(numRows, numCols, plotNum) 图表的整个绘图区域被分成 numRows 行和 numCols 列，此处表示一共是2行1列的图，此
# plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received') #(x,y,label)
# plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
# plt.yscale('log') #y用对数展示
# plt.xlabel('Date_received')
# plt.ylabel('Count')
# plt.legend() #显示图例（此处为两个label的说明）
#
# # 激活第2个 subplot
# plt.subplot(212)
# plt.bar(date_received_dt, buybydate['count']/couponbydate['count'])
# plt.xlabel('Date_received')
# plt.ylabel('Ratio(coupon used/coupon received)')
# plt.tight_layout() #tight_layout()调整子图之间的间隔来减少重复叠放。
#
# # plt.savefig('./pic/pic1.jpg')

# 如果Date=null & Date_received != null，该记录表示领取优惠券但没有使用，即负样本；
# 如果Date!=null & Date_received = null，则表示普通消费日期，表示没有领取优惠券也没有使用；
# 如果Date!=null & Date_received != null，则表示用优惠券消费日期，即正样本；

# 时间分类：
# Date!= null & Date_received == 'null' ——》 Date_received == 'null' ——》 y = -1  ——》没有领取优惠券，也没有使用；
# Date != 'null' & Date_received != null ——》 y = td  ——》领取优惠券了，时间间隔：用优惠券消费日期-领取优惠券日期
# Otherwise: y = 0 ——》领取优惠券了，没使用

def date_gap(row):
    if row['Date'] != -1 and row['Date_received'] != -1:
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        # if td <= pd.Timedelta(15, 'D'):
        # print(td.days)
        return td.days
    else:
        return -1



def is_purchase(row):
    if row['Date'] != -1:
        return 1
    return 0





def getWeekday(row):
    if row == '-1.0':
        return -1
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)



'''
user feature
'''
def get_user_feature(df):
    u = df[['User_id']].copy().drop_duplicates()

    # o_u_coupon_count : num of coupon received by user
    u1 = df[df['Date_received'] != -1][['User_id']].copy()
    u1['o_u_coupon_count'] = 1
    u1 = u1.groupby(['User_id'], as_index=False).count()

    # o_u_buy_count : times of user buy offline (with or without coupon)
    u2 = df[((df['Date'] != -1) & df['Action'] ==1)][['User_id']].copy()
    u2['o_u_buy_count'] = 1
    u2 = u2.groupby(['User_id'], as_index=False).count()

    # o_u_buy_with_coupon : times of user buy offline (with coupon)
    u3 = df[((df['Date'] != -1) & (df['Date_received'] != -1))][['User_id']].copy()
    u3['o_u_buy_with_coupon'] = 1
    u3 = u3.groupby(['User_id'], as_index=False).count()

    # o_u_merchant_count : num of merchant user bought from
    u4 = df[df['Date'] != -1][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace=True)
    u4 = u4.groupby(['User_id'], as_index=False).count()
    u4.rename(columns={'Merchant_id': 'o_u_merchant_count'}, inplace=True)
    # 用户点击次数
    u5 = df[df.Action == 0].groupby('User_id').size().reset_index(name='o_u_click_count')


    user_feature = pd.merge(u, u1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u5, on='User_id', how='left')



    user_feature['o_u_use_coupon_rate'] = user_feature['o_u_buy_with_coupon'].astype('float') / user_feature['o_u_coupon_count'].astype('float')
    user_feature['o_u_buy_with_coupon_rate'] = user_feature['o_u_buy_with_coupon'].astype('float') / user_feature['o_u_buy_count'].astype('float')
    user_feature = user_feature.fillna(-1)

    print(user_feature.columns.tolist())

    return user_feature

'''
merchant feature
'''
def get_merchant_feature(df):
    m = df[['Merchant_id']].copy().drop_duplicates()

    # o_m_coupon_count : num of coupon from merchant
    m1 = df[df['Date_received'] != -1][['Merchant_id']].copy()
    m1['o_m_coupon_count'] = 1
    m1 = m1.groupby(['Merchant_id'], as_index=False).count()

    # o_m_sale_count : num of sale from merchant (with or without coupon)
    m2 = df[((df['Date'] != -1) & df['Action'] ==1)][['Merchant_id']].copy()
    m2['o_m_sale_count'] = 1
    m2 = m2.groupby(['Merchant_id'], as_index=False).count()

    # o_m_sale_with_coupon : num of sale from merchant with coupon usage
    m3 = df[(df['Date'] != -1) & (df['Date_received'] != -1)][['Merchant_id']].copy()
    m3['o_m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index=False).count()

    #点击次数
    m4 = df[df.Action == 0].groupby('Merchant_id').size().reset_index(name='o_m_click_count')

    merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')


    merchant_feature['o_m_coupon_use_rate'] = merchant_feature['o_m_sale_with_coupon'].astype('float') \
                                              / merchant_feature['o_m_coupon_count'].astype('float')
    merchant_feature['o_m_sale_with_coupon_rate'] = merchant_feature['o_m_sale_with_coupon'].astype('float') \
                                                    / merchant_feature['o_m_sale_count'].astype('float')
    merchant_feature = merchant_feature.fillna(-1)

    print(merchant_feature.columns.tolist())
    return merchant_feature


'''
user-merchant cross feature
'''

def get_user_merchant_feature(df):
    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()

    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['o_um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um2 = df[((df['Date'] != -1) & df['Action'] ==1)][['User_id', 'Merchant_id']].copy()
    um2['o_um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um3 = df[df['Date_received'] != -1][['User_id', 'Merchant_id']].copy()
    um3['o_um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um4 = df[(df['Date_received'] != -1) & (df['Date'] != -1)][['User_id', 'Merchant_id']].copy()
    um4['o_um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    um5tmp = df[((df['Date_received'] != -1) & (df['Date'] != -1))][['User_id', 'Merchant_id','o_date_gap']].copy()
    um5tmp.replace(-1, np.nan, inplace=True)
    um5 = um5tmp.groupby(['User_id', 'Merchant_id'], as_index=False).mean()
    um5.rename(columns={'o_date_gap': 'o_um_mean_date_gap'}, inplace=True)

    #点击次数
    um6 = df[df.Action == 0].groupby(['User_id','Merchant_id']).size().reset_index(name='o_um_click_count')

    user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um5, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um6, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    user_merchant_feature['o_um_buy_rate'] = user_merchant_feature['o_um_buy_count'].astype('float')/user_merchant_feature['o_um_count'].astype('float')
    user_merchant_feature['o_um_coupon_use_rate'] = user_merchant_feature['o_um_buy_with_coupon'].astype('float')/user_merchant_feature['o_um_coupon_count'].astype('float')
    user_merchant_feature['o_um_buy_with_coupon_rate'] = user_merchant_feature['o_um_buy_with_coupon'].astype('float')/user_merchant_feature['o_um_buy_count'].astype('float')
    user_merchant_feature = user_merchant_feature.fillna(-1)

    print(user_merchant_feature.columns.tolist())
    return user_merchant_feature


def feature_combine_process(feature_base, feature):
    """
    combine user, merchant, and user_merchant feature
    """
    user_feature = get_user_feature(feature)
    # merchant_feature = get_merchant_feature(feature)
    # user_merchant_feature = get_user_merchant_feature(feature)

    feature = pd.merge(feature_base, user_feature, on='User_id', how='left')
    # feature = pd.merge(feature, merchant_feature, on='Merchant_id', how='left')
    # feature = pd.merge(feature, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    feature['o2o_u_buy_count_rate'] = feature['o_u_buy_count'].astype('float') / (feature['o_u_buy_count'].astype('float') + feature['u_buy_count'].astype('float'))
    feature['o2o_u_coupon_count_rate'] = feature['o_u_coupon_count'].astype('float') / (feature['o_u_coupon_count'].astype('float') + feature['u_coupon_count'].astype('float'))
    feature['o2o_u_buy_with_coupon_rate'] = feature['o_u_buy_with_coupon'].astype('float') / (feature['o_u_buy_with_coupon'].astype('float') + feature['u_buy_with_coupon'].astype('float'))



    feature = feature.fillna(-1)

    return feature



if __name__ == '__main__':
    dfoff = pd.read_csv('./o2o_data/ccf_offline_stage1_train.csv')
    dfon = pd.read_csv('./o2o_data/ccf_online_stage1_train.csv')
    dftest = pd.read_csv('./o2o_data/ccf_offline_stage1_test_revised.csv')
    dfoff_out = pd.read_csv('./o2o_data/offline_preprocess_out.csv')

    # nan --> -1
    dfon = dfon.fillna(-1)


    # discount process
    dfon = process_discount_data(dfon)

    # temporal process
    dfon['o_date_gap'] = dfon.apply(date_gap, axis=1)

    dfon['o_is_purchase'] = dfon.apply(is_purchase, axis=1)

    dfon['o_date_received_weekday'] = dfon['Date_received'].astype(str).apply(getWeekday)
    dfon['o_date_received_weekday_type'] = dfon['o_date_received_weekday'].apply(lambda x: 1 if x in [6, 7] else 0)  # weekday_type :  周六和周日为1，其他为0

    dfon['o_date_buy_weekday'] = dfon['Date'].astype(str).apply(getWeekday)
    dfon['o_date_buy_weekday_type'] = dfon['o_date_buy_weekday'].apply(lambda x: 1 if x in [6, 7] else 0)  # weekday_type :  周六和周日为1，其他为0


    # feature combine+ user +merchant +cross
    feature_base = dfoff_out.copy()
    df_out = feature_combine_process(feature_base, dfon)

    df_out.to_csv("./o2o_data/offline_online_preprocess_out.csv", index=False)

    df_out_na = df_out.replace(-1, np.nan)
    df_out_na.to_csv("./o2o_data/offline_online_preprocess_out_na.csv", index=False)