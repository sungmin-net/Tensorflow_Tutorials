# https://www.tensorflow.org/tutorials/estimator/linear
# 개요 : tf.estimator API 를 사용하여 로지스틱 회귀 모델 훈련
# 210120 에러 없이 마지막까지 실행은 되나, 마지막 그래프가 정상적으로 나오지 않음

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

# 타이타닉 데이터셋 불러오기

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head()) # print() added
print(dftrain.describe()) # print() added

print(dftrain.shape[0], dfeval.shape[0]) # print() added

print(dftrain.age.hist(bins = 20)) # print() added 이클립스에서는 차트가 도식되지 않는다..

print(dftrain.sex.value_counts().plot(kind='barh')) # print() added

print(pd.concat([dftrain, y_train], axis = 1).groupby('sex').survived.mean().plot(kind = 'barh').set_xlabel('% survive')) # print() added

# 모델을 위한 특성 공학(feature engineering)

# 기본 특성 열
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))
    
def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs = 1, shuffle = False)

ds = make_input_fn(dftrain, y_train, batch_size = 10)()
for feature_batch, label_batch in ds.take(1):
    print('특성 키:', list(feature_batch.keys()))
    print()
    print('클래스 배치:', feature_batch['class'].numpy())
    print()
    print('레이블 배치:', label_batch.numpy())
    
age_column = feature_columns[7]
print(tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()) # print() added

gender_column = feature_columns[0]
print(tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()) # print added

linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

# 도출된 특성 열
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size = 100)
derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns
        + derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)


pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

print(probs.plot(kind='hist', bins = 20, title = '예측 확률')) # print 추가. 그래프는 나오지 않음

from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('fpr(false positive rate)')
plt.ylabel('tpr(true positive rate)')
plt.xlim(0,)
plt.ylim(0,)
plt.show() # added. 그런데 모양이 너무다르다..
