import numpy
numpy.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape, Dropout
from keras.layers import Merge
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

import pickle
from future import *

def embed_features(X, saved_embeddings_fname):
    # f_embeddings = open("embeddings_shuffled.pickle", "rb")
    f_embeddings = open(saved_embeddings_fname, "rb")
    embeddings = pickle.load(f_embeddings)

    index_embedding_mapping = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    X_embedded = []

    (num_records, num_features) = X.shape
    for record in X:
        embedded_features = []
        for i, feat in enumerate(record):
            feat = int(feat)
            if i not in index_embedding_mapping.keys():
                embedded_features += [feat]
            else:
                embedding_index = index_embedding_mapping[i]
                embedded_features += embeddings[embedding_index][feat].tolist()

        X_embedded.append(embedded_features)

    return numpy.array(X_embedded)


def split_features(X):
    X_list = []

    cate_1 = X[..., [0]]
    X_list.append(cate_1)

    cate_2 = X[..., [1]]
    X_list.append(cate_2)

    item_id = X[..., [2]]
    X_list.append(item_id)

#    year = X[..., [3]]
#    X_list.append(year)

#    month = X[..., [4]]
#    X_list.append(month)

#    day = X[..., [5]]
#    X_list.append(day)

    week_day = X[..., [6]]
    X_list.append(week_day)

    #discount = X[..., [7]]
    #X_list.append(discount)

    #cont = X[..., [8]]
    cont = X[..., [7,8,9,10]]
    X_list.append(cont)

    return X_list


class Model(object):

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result


class LinearModel(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(LinearModel, self).__init__()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(X_train, numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class RF(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(RF, self).__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=True, max_depth=35, min_samples_split=2,
                                         min_samples_leaf=1)
        self.clf.fit(X_train, numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class SVM(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(SVM, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.__normalize_data()
        self.clf = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001,
                       C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

        self.clf.fit(self.X_train, numpy.log(self.y_train))
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def __normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

    def guess(self, feature):
        return numpy.exp(self.clf.predict(feature))


class XGBoost(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(XGBoost, self).__init__()
        dtrain = xgb.DMatrix(X_train, label=numpy.log(y_train))
        evallist = [(dtrain, 'train')]
        param = {'nthread': -1,
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = 3000
        self.bst = xgb.train(param, dtrain, num_round, evallist)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return numpy.exp(self.bst.predict(dtest))


class HistricalMedian(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(HistricalMedian, self).__init__()
        self.history = {}
        self.feature_index = [1, 2, 3, 4]
        for x, y in zip(X_train, y_train):
            key = tuple(x[self.feature_index])
            self.history.setdefault(key, []).append(y)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = numpy.array(features)
        features = features[:, self.feature_index]
        guessed_sales = [numpy.median(self.history[tuple(feature)]) for feature in features]
        return numpy.array(guessed_sales)


class KNN(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(KNN, self).__init__()
        self.normalizer = Normalizer()
        self.normalizer.fit(X_train)
        self.clf = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance', p=1)
        self.clf.fit(self.normalizer.transform(X_train), numpy.log(y_train))
        print("Result on validation data: ", self.evaluate(self.normalizer.transform(X_val), y_val))

    def guess(self, feature):
        return numpy.exp(self.clf.predict(self.normalizer.transform(feature)))


class NN_with_EntityEmbedding(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(NN_with_EntityEmbedding, self).__init__()
        self.nb_epoch = 4
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model_v2()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model_v2(self):
        models = []

        model_cate1 = Sequential()
        model_cate1.add(Embedding(15, 4, input_length=1))
        model_cate1.add(Reshape(target_shape=(4,)))
        models.append(model_cate1)

        model_cate2 = Sequential()
        model_cate2.add(Embedding(74, 5, input_length=1))
        model_cate2.add(Reshape(target_shape=(5,)))
        models.append(model_cate2)

        model_item =  Sequential()
#        model_item.add(Embedding(2307, 10, input_length=1))
        model_item.add(Embedding(9681, 12, input_length=1))
#        model_item.add(Reshape(target_shape=(10,)))
        model_item.add(Reshape(target_shape=(12,)))
        models.append(model_item)

#        model_year = Sequential()
#        model_year.add(Embedding(2, 1, input_length=1))
#        model_year.add(Reshape(target_shape=(1,)))
#        models.append(model_year)

#        model_month = Sequential()
#        model_month.add(Embedding(12, 5, input_length=1))
#        model_month.add(Reshape(target_shape=(5,)))
#        models.append(model_month)

#        model_day = Sequential()
#        model_day.add(Embedding(31, 8, input_length=1))
#        model_day.add(Reshape(target_shape=(8,)))
#        models.append(model_day)

        model_week_day = Sequential()
        model_week_day.add(Embedding(7, 4, input_length=1))
        model_week_day.add(Reshape(target_shape=(4,)))
        models.append(model_week_day)

#        model_discount = Sequential()
#        model_discount.add(Embedding(4, 3, input_length=1))
#        model_discount.add(Reshape(target_shape=(3,)))
#        models.append(model_discount)


        model_continue = Sequential()
        model_continue.add(Dense(4, input_dim=4))
        models.append(model_continue)

        #model_continue = Sequential()
        #model_continue.add(Dense(1, input_dim=1))
        #models.append(model_continue)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_absolute_error', optimizer='adam')
        #self.model.compile(loss='mse', optimizer='adam')


    def __build_keras_model(self):
        models = []

        model_store = Sequential()
        model_store.add(Embedding(8847, 12, input_length=1))
        model_store.add(Reshape(target_shape=(10,)))
        models.append(model_store)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 6, input_length=1))
        model_dow.add(Reshape(target_shape=(6,)))
        models.append(model_dow)

        model_promo = Sequential()
        model_promo.add(Dense(1, input_dim=1))
        models.append(model_promo)

        model_year = Sequential()
        model_year.add(Embedding(3, 2, input_length=1))
        model_year.add(Reshape(target_shape=(2,)))
        models.append(model_year)

        model_month = Sequential()
        model_month.add(Embedding(12, 6, input_length=1))
        model_month.add(Reshape(target_shape=(6,)))
        models.append(model_month)

        model_day = Sequential()
        model_day.add(Embedding(31, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)

        model_germanstate = Sequential()
        model_germanstate.add(Embedding(12, 6, input_length=1))
        model_germanstate.add(Reshape(target_shape=(6,)))
        models.append(model_germanstate)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(lr=0.0001, loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       nb_epoch=self.nb_epoch, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)


class NN(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        super(NN, self).__init__()
        self.nb_epoch = 10
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(numpy.max(numpy.log(y_train)), numpy.max(numpy.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(Dense(1000, init='uniform', input_dim=1183))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(lr=0.0001, loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, self._val_for_fit(y_train),
                       validation_data=(X_val, self._val_for_fit(y_val)),
                       nb_epoch=self.nb_epoch, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
