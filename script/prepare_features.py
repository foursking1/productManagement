import pickle
from datetime import datetime
from datetime import date
from sklearn import preprocessing
import numpy as np
import random
import math
from config import date_file_path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
random.seed(42)

prepare_data_path = date_file_path['prepare_dump_path']
with open(prepare_data_path, 'rb') as f:
    df_train = pickle.load(f)


def feature_engineering(df_train):
    scaler =  StandardScaler()
    df_train['date'] = df_train['time']
    df_train['time'] = df_train['time'].map(lambda e: datetime.strptime(e, '%Y/%m/%d'))
    df_train['year'] = df_train['time'].map(lambda x: x.year)
    df_train['month'] = df_train['time'].map(lambda x: x.month)
    df_train['day'] = df_train['time'].map(lambda x: x.day)

    df_train['retail_price'] = df_train['retail_price'].map(lambda x: math.log(x))
    df_train['retail_price'] = scaler.fit_transform(df_train['retail_price'])
    df_train['total_uv'] = scaler.fit_transform(df_train['total_uv'])
    df_train['uv'] = scaler.fit_transform(df_train['uv'])

    df_train['week_day'] = df_train['week_day'].astype(int)

    df_train, df_test = df_train[df_train['time'] <  date(2017,7,2)], df_train[df_train['time'] >= date(2017,7,2)]
    print "train size %d" % len(df_train.index)
    print "test size %d" % len(df_test.index)

    df_train_X = df_train[['cate_1', 'cate_2', 'skuid', 'year', 'month', 'day', 'week_day', 'discount', 'retail_price', 'total_uv', 'uv']]
    df_train_Y = df_train['amount']

    on_sale_id = df_train_X['skuid'].values
    on_sale_cate_name = df_train_X['cate_2'].values

    df_test = df_test[df_test['skuid'].isin(on_sale_id)]
    df_test = df_test[df_test['cate_2'].isin(on_sale_cate_name)]

    with open('test_df.pickle', 'wb') as f:
        pickle.dump(df_test, f, -1)

    df_test_X = df_test[['cate_1', 'cate_2', 'skuid', 'year', 'month', 'day', 'week_day', 'discount', 'retail_price', 'total_uv', 'uv']]
    df_test_Y = df_test['amount']
    return df_train_X.values, df_train_Y.values, df_test_X.values, df_test_Y.values

print df_train.head()
train_data_X, train_data_y, test_data_X, test_data_y = feature_engineering(df_train)
print("Number of train datapoints: ", len(train_data_y))
print("Number of test datapoints: ", len(test_data_y))

print(min(train_data_y), max(train_data_y))
print(min(test_data_y), max(test_data_y))
#

full_X = train_data_X
full_X = np.array(full_X)
train_data_X = np.array(train_data_X)
test_data_X = np.array(test_data_X)
les = []


for i in range(8):
    le = LabelEncoder()
    le.fit(full_X[:, i])
    print len(le.classes_)
    les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])
    train_data_X[:, i].astype(int)

    test_data_X[:, i] = le.transform(test_data_X[:, i])
    test_data_X[:, i].astype(int)

with open('les.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

train_data_X = train_data_X
train_data_y = np.array(train_data_y)
print len(train_data_X)

with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

test_data_X = test_data_X
test_data_y = np.array(test_data_y)
print len(test_data_X)

with open('feature_test_data.pickle', 'wb') as f:
    pickle.dump((test_data_X, test_data_y), f, -1)
    print(test_data_X[0], test_data_y[0])
