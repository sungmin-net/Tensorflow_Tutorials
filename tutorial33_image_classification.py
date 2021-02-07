# https://www.tensorflow.org/tutorials/images/classification
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
im = PIL.Image.open(str(roses[0])) # 'im =' added
#im.show() # added

im = PIL.Image.open(str(roses[1])) # 'im =' added
#im.show() # added

tulips = list(data_dir.glob('tulips/*'))
im = PIL.Image.open(str(tulips[0])) # 'im = ' added
#im.show() #  added

im = PIL.Image.open(str(tulips[1])) # 'im = ' added
#im.show() #  added

# keras.preprocessing 을 사용하여 로드
# 데이터세트 만들기
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2,
        subset = 'training', seed = 123, image_size = (img_height, img_width),
        batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2,
        subset = 'validation', seed = 123, image_size = (img_height, img_width),
        batch_size = batch_size)

class_names = train_ds.class_names
print(class_names)

# 데이터 시각화하기
plt.figure(figsize = (10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
#plt.show() # added

for image_batch, labels_batch in train_ds :
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# 성능을 높이도록 데이터셋 구성하기
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# 데이터 표준화
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y : (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in '[0, 1]'.
print(np.min(first_image), np.max(first_image))

# 모델 만들기
num_classes = 5
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes)
])

# 모델 컴파일
model.compile(optimizer = 'adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])

# 모델 요약
model.summary()

# 모델 훈련
epochs = 10
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)

# 훈련 결과 시각화하기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize = (8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()

# 데이터 증강
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', 
            input_shape = (img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

# 데이터가 부족하여 과적합이 발생하면 데이터 증강을 사용
# 동일한 이미지에서 데이터 증강을 여러번 적용하여 몇가지 증강된 예제가 어떻게 보이는지 시각화
plt.figure(figsize = (10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')
        
# 드롭아웃 - 과적합을 줄이는 다른 기술
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes)
])

# 모델 컴파일 및 훈련
model.compile(optimizer = 'adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
        metrics = ['accuracy'])
model.summary()

epochs = 15
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)

# 훈련 결과 시각화하기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize = (8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training and Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()

# 새로운 데이터로 예측하기
# 모델을 사용하여 훈련 또는 검증 세트에 포함되지 않은 이미지를 분류
sunflower_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg'
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin = sunflower_url)
img = keras.preprocessing.image.load_img(sunflower_path, target_size = (img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score))
)
