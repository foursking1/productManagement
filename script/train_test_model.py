import pickle
import dill
import numpy
numpy.random.seed(123)
from models import *
from sklearn.preprocessing import OneHotEncoder
import sys
sys.setrecursionlimit(10000)

train_ratio = 0.9
shuffle_data = True
one_hot_as_input = False
embeddings_as_input = False
save_embeddings = True
save_models = True
saved_embeddings_fname = "embeddings.pickle"  # Use plot_embeddings.ipynb to create

f_train = open('feature_train_data.pickle', 'rb')
(X, y) = pickle.load(f_train)

f_test = open('feature_test_data.pickle', 'rb')
(test_X, test_y) = pickle.load(f_test)

num_records = len(X)
train_size = int(train_ratio * num_records)

if shuffle_data:
    print("Using shuffled data")
    sh = numpy.arange(X.shape[0])
    numpy.random.shuffle(sh)
    X = X[sh]
    y = y[sh]

if embeddings_as_input:
    print("Using learned embeddings as input")
    X = embed_features(X, saved_embeddings_fname)

if one_hot_as_input:
    print("Using one-hot encoding as input")
    enc = OneHotEncoder(sparse=False)
    enc.fit(X)
    X = enc.transform(X)

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]


def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]

X_train, y_train = sample(X_train, y_train, 600000)  # Simulate data sparsity
#X_train, y_train = sample(X_train, y_train, 4000)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

models = []

print("Fitting NN_with_EntityEmbedding...")
for i in range(5):
    models.append(NN_with_EntityEmbedding(X_train, y_train, X_val, y_val))

# print("Fitting NN...")
# for i in range(5):
#     models.append(NN(X_train, y_train, X_val, y_val))

# print("Fitting RF...")
# models.append(RF(X_train, y_train, X_val, y_val))

# print("Fitting KNN...")
# models.append(KNN(X_train, y_train, X_val, y_val))

# print("Fitting XGBoost...")
# models.append(XGBoost(X_train, y_train, X_val, y_val))


if save_embeddings:
    model = models[0].model
    weights = model.get_weights()
    print model.summary()

    for index, weight in enumerate(weights):
        if len(weight) > 0:
            print index, weight[0].shape

    cate_1 = weights[0]
    cate_2 = weights[2]
    skuid = weights[4]
    month_embedding = weights[8]
    day_embedding = weights[10]
    week_day_embedding = weights[12]
    discount_embedding = weights[14]
    with open(saved_embeddings_fname, 'wb') as f:
        pickle.dump([cate_1, cate_2, skuid,
                    month_embedding, day_embedding, week_day_embedding, discount_embedding], f, -1)

if save_models:
    with open('models.pickle', 'wb') as f:
        #pickle.dump(models, f)
        dill.dump(models, f)


def evaluate_models(models, X, y):
    assert(min(y) > 0)
    guessed_sales = numpy.array([model.guess(X) for model in models])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = numpy.absolute((y - mean_sales) / y)
    result = numpy.sum(relative_err) / len(y)
    return result

print("Evaluate combined models...")
print("Training error...")
r_train = evaluate_models(models, X_train, y_train)
print(r_train)

print("Validation error...")
print len(y_val)
r_val = evaluate_models(models, X_val, y_val)
print(r_val)

print("next month error...")
print len(test_y)
r_test = evaluate_models(models, test_X, test_y)
print(r_test)

f = open('test_df.pickle', 'rb')
df_test = pickle.load(f)
guessed_sales = numpy.array([model.guess(test_X) for model in models])
mean_sales = guessed_sales.mean(axis=0)
df_test['pred'] = mean_sales
print df_test.head()

with open('result_df.pickle', 'wb') as f:
    pickle.dump(df_test, f, -1)





