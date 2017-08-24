#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from config import date_file_path
import pickle


uv_path = '../data/uv.csv'
item_uv_path = '../data/item_20160801_20170731.csv'
data_path = '../data/data.csv'


item_uv_list = []
with open(item_uv_path) as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.split(",")
        item_uv_list.append({'item_id': line[1], 'time': line[3], 'uv': line[5], 'new_uv': line[7]})
    print item_uv_list[0]

df_item_uv = pd.DataFrame(item_uv_list)
print df_item_uv.head()

week_day_loop_table = {"星期一": 1, "星期二": 2, "星期三": 3, "星期四": 4, "星期五": 5, "星期六": 6, "星期日": 0,}

uv_list = []
with open(uv_path) as f:
    for index, line in enumerate(f.readlines()):
        if index == 0:
            continue
        line = line.strip('\n')
        line = line.split(",")
        uv_list.append({'time': line[0].split(" ")[0], 'week_day': week_day_loop_table[line[1]], 'total_uv': line[5]})
    print uv_list[0]

df_uv = pd.DataFrame(uv_list)
print df_uv.head()

le = preprocessing.LabelEncoder()
df_train = pd.read_csv(data_path)

#df_train['cate_1'] = le.fit_transform(df_train['phy_category1_name'])
#df_train['cate_2'] = le.fit_transform(df_train['phy_category2_name'])
#df_train['discount'] = le.fit_transform(df_train['折扣范围'])

df_train['cate_1'] = df_train['phy_category1_name']
df_train['cate_2'] = df_train['phy_category2_name']
df_train['discount'] = df_train['折扣范围']

df_train['time'] = df_train['日期']

df_train = df_train[['cate_1', 'cate_2', 'item_id', 'time', 'retail_price', 'discount', 'user', 'amount', 'money', 'newuser','newamount','newmoney']]

# convert data type
df_train['item_id'] = df_train['item_id'].astype("int")
df_item_uv['item_id'] = df_item_uv['item_id'].astype("int")

print df_train.head()
print 'init size: %d' % (len(df_train.index))

df_train = pd.merge(df_train, df_uv, how='left', on='time')
df_train = pd.merge(df_train, df_item_uv, how='inner', on=['item_id', 'time'])

#print df_item_uv.dtypes
#print df_train.dtypes
print df_train.head()
print 'total size: %d' % (len(df_train.index))

with open(date_file_path['prepare_dump_path'], 'wb') as f:
    pickle.dump(df_train, f)