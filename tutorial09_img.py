# https://www.tensorflow.org/tutorials/load_data/images?hl=ko

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin = dataset_url, fname = 'flower_photos', untar = True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))#.show() # 추가함 
PIL.Image.open(str(roses[1]))#.show()


# keras.preprocessing 을 사용하여 로드

# 로더 매개변수
batch_size = 32
img_height = 180
img_width = 180

# 유효성 검사 분할. 80% 훈련, 20% 유효성 검사
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2,
        subset = "training", seed = 123, image_size = (img_height, img_width), 
        batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2,
        subset = "validation", seed = 123, image_size = (img_height, img_width),
        batch_size = batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
#plt.show() # 추가함        

for image_batch, labels_batch in train_ds :
    print(image_batch.shape)
    print(labels_batch.shape)
    break;

from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in '[0, 1]'.
print(np.min(first_image), np.max(first_image))

# 성능을 위한 데이터 세트 구성

AUTOTUNE = tf.data.experimental.AUTOTUNE # 에러남 -> https://ahnjg.tistory.com/32 -> .experimental 추가
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)


# 모델 훈련
num_classes = 5
model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(num_classes)            
    ])

model.compile(optimizer = 'adam', loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])

model.fit(train_ds, validation_data = val_ds, epochs = 3)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle = False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration = False)

for f in list_ds.take(5):
    print(f.numpy())

class_names = np.array(sorted([item.name for item in data_dir.glob('*')
        if item.name != 'LICENSE.txt']))

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
    # convert the path to a list of path compoments
    parts = tf.strings.split(file_path, os.path.sep)
    # the second to last is the class-directory
    one_hot = parts[-2] == class_names
    # integer encode the label
    return tf.argmax(one_hot)

def decode_img(img):
    # convert the compressed string to a 3D unit8 tensor
    img = tf.image.decode_jpeg(img, channels = 3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# set 'num_parallel_calls' so multiple images are loaded / processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls = AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls = AUTOTUNE)

'''
train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
'''

for image, label in train_ds.take(1):
    print("Image shape : ", image.numpy().shape)
    print("Label : ", label.numpy()) # why label : 0? not 3?

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize = (10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis('off')

plt.show() # 추가함. 잘 나옴.

model.fit(train_ds, validation_data = val_ds, epochs = 3)

(train_ds, val_ds, test_ds), metadata = tfds.load("tf_flowers", 
        split = ['train[:80%]','train[80%:90%]', 'train[90%:]'], with_info = True, 
        as_supervised = True)
         
num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str
image, label = next(iter(train_ds))
_ = plt.imshow(image) # 이건 무슨 코드인가.. _= 를 없애봄 > 다시 살림
_ = plt.title(get_label_name(label))

'''
plt.imshow(image)
plt.title(get_label_name(label))
'''
plt.show() # 추가함

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)
