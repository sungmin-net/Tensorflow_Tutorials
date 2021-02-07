# https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 데이터 다운로드 및 준비
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0 ~ 1 사이로 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 합성곱 층 만들기
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

# 모델 컴파일과 훈련
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)