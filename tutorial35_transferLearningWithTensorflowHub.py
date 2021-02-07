# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

import matplotlib.pylab as plt
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import layers

# ImageNet 분류기
# 분류기 다운로드
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape = IMAGE_SHAPE + (3,))])

# 싱글 이미지 실행시키기
import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg', 
        'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
#grace_hopper.show() # .show() added

grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape) # print() added

# 차원 배치를 추가하고 이미지를 모델에 통과시킴
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape) # print() added

# 로지트는 1001 요소 벡터. 이는 이미지에 대한 각각의 클래스 확률을 계산
# 탑 클래스인 ID는 최대값을 알 수 있음.......무슨 소리야? -_-
predicted_class = np.argmax(result[0], axis = -1)
print(predicted_class) # print() added  

# 클래스 ID 를 예측하고, ImageNet 라벨을 불러오고, 예측을 해독
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction : " + predicted_class_name.title())
#plt.show() # added

# 간단한 전이 학습
data_root = tf.keras.utils.get_file('flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar = True)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size = IMAGE_SHAPE)

for image_batch, label_batch in image_data:
    print("Image batch shape : ", image_batch.shape)
    print("Label batch shape : ", label_batch.shape)
    break;

# 이미지 배치에 대한 분류기 실행
result_batch = classifier.predict(image_batch)
print(result_batch.shape) # print() added

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis = -1)]
print(predicted_class_names) # print() added

# 얼마나 많은 예측들이 이미지에 맞는 지 검토
plt.figure(figsize = (10, 9))
plt.subplots_adjust(hspace = 0.5)
for n  in range(30):
    plt.subplot(6, 6, n + 1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
#plt.show() # added

# 헤드리스 모델을 다운로드
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

# 특성 추출기 만들기
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape = (224, 224, 3))
# 각각의 이미지마다 길이가 1280인 벡터가 반환됨
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
# 특성 추출기 계층에 있는 변수들을 굳히면, 학습은 오직 새로운 분류 계층만 변형시킬 수 있음
feature_extractor_layer.trainable = False

# 분류 head를 붙이기
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes, activation = 'softmax')
])

model.summary()
predictions = model(image_batch)
print(predictions.shape) # print() added

# 모델 학습 - 학습 과정 환경을 설정하기 위해 컴파일 사용
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy',
        metrics = ['acc'])

# 모델을 학습시키기 위해 .fit 메소드 사용
class CollectBatchStates(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    
    def on_train_batch_end(self, batch, logs = None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)
batch_stats_callback = CollectBatchStates()
history = model.fit_generator(image_data, epochs = 2, steps_per_epoch = steps_per_epoch,
        callbacks = [batch_stats_callback])

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_losses)
#plt.show() # added

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training steps")
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)
#plt.show() # added

# 예측 확인 - 클래스의 이름들로 정렬된 리스트의 첫번째를 얻는다.
class_names = sorted(image_data.class_indices.items(), key = lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

# 모델을 통해 이미지 배치를 실행(이미지를 모델에 통과 시키기), 인덱스들을 클래스 이름으로 바꾸기
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis = -1)
predicted_label_batch = class_names[predicted_id]

# 결과 계획
label_id = np.argmax(label_batch, axis = -1)
plt.figure(figsize = (10, 9))
plt.subplots_adjust(hspace = 0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    color = 'green' if predicted_id[n] == label_id[n] else 'red'
    plt.title(predicted_label_batch[n].title(), color = color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (green : correct, red : incorrect)")
plt.show()# added

# 모델 내보내기
import time
t = time.time()
export_path = '/tmp/saved_models/{}'.format(int(t))
model.save(export_path, save_format = 'tf')

print(export_path) # print() added
reloaded = tf.keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
print(abs(reloaded_result_batch - result_batch).max())