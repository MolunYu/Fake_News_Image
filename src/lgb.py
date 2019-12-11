import lightgbm as lgb
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# data prepare
train_dataset = pd.read_csv('../data/lgb/train_feature.csv')
x_train = train_dataset.iloc[:, :5].values
y_train = train_dataset.iloc[:, 5].values
d_train = lgb.Dataset(x_train, label=y_train)

test_dataset = pd.read_csv('../data/lgb/test_feature.csv')
x_test = test_dataset.iloc[:, :5].values
y_test = test_dataset.iloc[:, 5].values

# Hyper-parameters
params = {
    'learning_rate': 0.01,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'sub_feature': 0.5,
    'num_leaves': 128,
    'min_data_in_leaf': 50
}

num_boosts = 1000
threshold = 0.5

clf = lgb.train(params, d_train, num_boosts)
y_pred = clf.predict(x_test)
for i in range(len(y_pred)):
    if y_pred[i] >= threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
clf.save_model("../data/lgb/{}.txt".format(time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())))

accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test)
recalls = recall_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test)

print("accuracy: ", accuracy)
print("precision: ", precision)
print("recalls: ", recalls)
print("f1: ", f1)

