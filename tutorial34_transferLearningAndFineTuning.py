# https://www.tensorflow.org/tutorials/images/transfer_learning

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
keras = tf.keras

# 데이터 전처리
# 데이터 다운로드
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', 
        split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info = True, 
        as_supervised = True,)
 
print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label)) 
#plt.show() # added

# 데이터포멧
IMG_SIZE = 160 # 모든 이미지는 160x160으로 크기가 조정됨
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# map 함수를 사용하여 dataset 의 각 항목에 이 함수를 적용
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# 데이터를 섞고 일괄 처리
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# 데이터 검사
for image_batch, label_batch in train_batches.take(1):
    pass # 이게 무슨 짓?

print(image_batch.shape)

# 사전 훈련된 컴볼루션 네트워크로부터 기본 모델 생성하기

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# 사전 훈련된 모델 MobileNet V2 에서 기본 모델을 생성
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False,
        weights = 'imagenet')
# 특징 추출기는 각 160 x 160 x 3 이미지를 5 x 5 x 1280 개의 특징 블록으로 변환
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# 특징 추출
# 컨볼루션 베이스 모델 고정하기
base_model.trainable = False

# 기본 모델 아키텍처를 살펴보기
base_model.summary()

# 분류 층을 맨 위에 추가
# 특징을 이미지 한개당 1280개의 요소 벡터로 변환하여 5x5 공간 위치에 대한 평균을 구함
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# tf.keras.layers.Dense 층을 사용하여 특징을 이미지당 단일 예측으로 변환
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# tf.keras.Sequential 모델을 사용하여 특징 추출기와 두 층을 쌓기
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# 모델 컴파일
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
model.summary()
print(len(model.trainable_variables)) # print() added

# 모델 훈련
initial_epochs = 10
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss : {:.2f}".format(loss0))
print("initial accuracy : {:.2f}".format(accuracy0))

history = model.fit(train_batches, epochs = initial_epochs, validation_data = validation_batches)

# 학습 곡선
# MobileNet V2 기본 모델을 고정된 특징 추출기로 사용했을 때의 학습 및 검증 정확도 / 손실의 학습 곡선
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = "Training Accuracy")
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

base_model.trainable = True
# 기본 모델에 몇 개의 층이 있는 지 확인
print("Number of layers in the base model : ", len(base_model.layers))

# 해당 층 이후부터 미세 조정
fine_tune_at = 100

# 'fine_tune_at' 층 이전의 모든 층을 고정
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# 모델 컴파일
# 훨씬 더 낮은 학습 비율로 모델 컴파일
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate / 10),
        metrics = ['accuracy'])
model.summary()
print(len(model.trainable_variables)) # print() added

# 모델 훈련 계속하기
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_batches, epochs = total_epochs, initial_epoch = history.epoch[-1],
        validation_data = validation_batches)

# 미세 조정 이후 모델은 98% 정확도에 도달
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start Fine Tuning')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start Fine Tuning')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()