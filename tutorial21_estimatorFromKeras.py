#210121_https://www.tensorflow.org/tutorials/estimator/keras_model_to_estimator

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# 간단한 케라스 모델(완전히 연결된 네트워크 - 다층 퍼셉트론) 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu', input_shape = (4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
        optimizer = 'adam')
model.summary()

# 입력 함수 만들기
def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split = split, as_supervised = True)
    dataset = dataset.map(lambda features, labels: ({'dense_input' : features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset

# 입력 함수 확인
for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)

# tf.keras.model 을 추정기로 변환하기

import tempfile
model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model = model, model_dir = model_dir)

keras_estimator.train(input_fn = input_fn, steps = 500)
eval_result = keras_estimator.evaluate(input_fn = input_fn, steps = 10)
print("평가 결과 : {}".format(eval_result))
     
