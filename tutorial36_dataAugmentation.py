# https://www.tensorflow.org/tutorials/images/data_augmentation

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 데이터셋 다운로드
(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',
        split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info = True, 
        as_supervised = True)

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
#plt.show() # added

# keras 전처리 레이어 사용하기
# 크기 및 배율 조정하기 - 픽셀 값을 [-1,1]로 하길 원할 경우, Rescaling(1./127.5, offset = -1)을 작성
IMG_SIZE = 100
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

result = resize_and_rescale(image)
_ = plt.imshow(result)
#plt.show() # added

# 픽셀이 [0-1]에 있는 지 확인
print("Min and max pixel values : ", result.numpy().min, result.numpy().max())

# 데이터 증강 - 전처리 레이어를 사용하여 동일한 이미지에 반복 적용
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# add the image to a batch
image = tf.expand_dims(image, 0)
plt.figure(figsize = (10, 10))
for i in range(9) :
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
#plt.show() # added

# 전처리 레이어를 사용하는 두 가지 옵션
# 옵션1 : 전처리 레이어를 모델의 일부로 만들기
model = tf.keras.Sequential([
    resize_and_rescale, data_augmentation,
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    # Rest of your model
    # 참고 : 데이터 증강은 테스트할 때 비활성화되므로 입력이미지는 model.fit(model.evaluate 또는
    # model.predict 가 아님) 호출 중에만 증강된다.    
])

# 옵션2 : 데이터세트에 전처리 레이어 적용하기
aug_ds = train_ds.map(lambda x, y : (resize_and_rescale(x, training = True), y))
# 참고 : 이 경우 전처리 레이어는 model.save 호출 호출할 때 모델과 함께 내보내지지 않음. 저장 전 이
# 레이어를 모델에 연결하거나 서버측에서 다시 구현해야 함. 훈련 후 내보내기 전에 전처리 레이어 연결 가능

# 데이터셋에 전처리 레이어 적용하기
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
def prepare(ds, shuffle = False, augment = False):
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y : (resize_and_rescale(x), y), num_parallel_calls = AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    
    # Batch all datasets
    ds = ds.batch(batch_size)
    
    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y : (data_augmentation(x, training = True), y), 
                num_parallel_calls = AUTOTUNE)
    
    # Use buffered prefetching on all datasets
    return ds.prefetch(buffer_size = AUTOTUNE)

train_ds = prepare(train_ds, shuffle = True, augment = True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# 모델 훈련하기
model = tf.keras.Sequential([
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

model.compile(optimizer = 'adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
        metrics=['accuracy'])

epochs = 5
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)
loss, acc = model.evaluate(test_ds)
print("Accuracy : ", acc)

# 사용자 정의 데이터 증강
# 두 가지 방법 : 1. layers.lambda 레이어 생성(간결한 코드), 2. 서브 클래스 생성(제어력 증가)
def random_invert_img(x, p = 0.5):
    if tf.random.uniform([]) < p:
        x = (255 - x)
    else :
        x
    return x;


def random_invert(factor=0.5):
    return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()

plt.figure(figsize = (10, 10))
for i in range(9):
    augmented_image = random_invert(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype("uint8"))
    plt.axis('off')
#plt.show() # added

# 서브클래스 생성을 통해 사용자 정의 레이어 구현
class RandomInvert(layers.Layer):
    def __init__(self, factor = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    
    def call(self, x):
        return random_invert_img(x)
_ = plt.imshow(RandomInvert()(image)[0])
#plt.show() # added

# tf.image 사용하기
# 꽃 데이터셋은 이전에 데이터 증강으로 구성되었으므로 다시 가져와서 새로 시작
(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers', 
        split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info = True,
        as_supervised = True)

# 작업할 이미지를 검색
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
#plt.show() # added

# 다음 함수를 사용하여 원본 이미지와 증강 이미지를 나란히 시각화하고 비교
def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)
    
    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)

# 데이터 증강
# 이미지 뒤집기 : 이미지를 수직 또는 수평으로 뒤집기
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)
#plt.show()

# 이미지를 회색조로 만들기
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()
#plt.show() # added

# 이미지 포화시키기 - 채도 계수를 제공하여 이미지를 포화
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)
#plt.show() # added

# 이미지 밝기 변경하기 : 밝기 계수를 제공하여 이미지의 밝기를 변경
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)
#plt.show() # added

# 이미지 중앙 자르기 : 이미지를 중앙에서 원하는 이미지 부분까지 자름
cropped = tf.image.central_crop(image, central_fraction = 0.5)
visualize(image, cropped)
#plt.show() # added

# 이미지 회전하기 : 90도 회전
rotated = tf.image.rot90(image)
visualize(image, rotated)
plt.show() # added

# 데이터셋에 증강 적용
# 이전과 마찬가지로 Dataset.map 을 사용하여 데이터 증강을 데이터셋에 적용
def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label

def augment(image, label):
    image, label = resize_and_rescale(image, label)
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Random crop back to the original size
    image = tf.image.random_crop(image, size = [IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta = 0.5) # Random brightness
    image = tf.clip_by_value(image, 0, 1)
    return image, label

# 데이터셋 구성하기
train_ds = (train_ds.shuffle(1000).map(augment, num_parallel_calls = AUTOTUNE).batch(batch_size)
        .prefetch(AUTOTUNE))

val_ds = (val_ds.map(resize_and_rescale, num_parallel_calls = AUTOTUNE).batch(batch_size)
        .prefetch(AUTOTUNE))

test_ds = (test_ds.map(resize_and_rescale, num_parallel_calls = AUTOTUNE).batch(batch_size)
        .prefetch(AUTOTUNE))

# 이전과 같이 모델 훈련
model = tf.keras.Sequential([
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

model.compile(optimizer = 'adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
        metrics=['accuracy'])

epochs = 5
history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)
loss, acc = model.evaluate(test_ds)
print("Accuracy : ", acc)