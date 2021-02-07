# https://www.tensorflow.org/tutorials/generative/autoencoder

# 라이브러리 가져오기
import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# 데이터셋 로드
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

# 첫번쩨 예 : 기본 autoencoder
# 두 개의 Dense 레이어로 오토인코더를 정의. 이미지를 64차원 잠재 벡터로 압축하는 encoder 와
# 잠재 공간에서 원본 이미지를 재구성하는 decoder.
latent_dim = 64
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
            ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
            ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# x_train 을 입력과 대상으로 사용하여 모델 훈련. encoder는 784차원에서 잠재 공간을 압축하는 방법을
# 배우고, decoder 는 원본 이미지를 재구성하는 방법을 배움
autoencoder.fit(x_train, x_train, epochs = 10, shuffle = True, validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show() - 잘 나옴

# 두번째 예: 이미지 노이즈 제거
# 데이터셋 다시 가져오기
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[... , tf.newaxis]
x_test = x_test[... , tf.newaxis]

print(x_train.shape)

# 이미지에 임의의 노이즈를 추가
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min = 0., clip_value_max = 1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min = 0., clip_value_max = 1.)

# 노이즈가 있는 이미지를 플롯
n = 10
plt.figure(figsize=(20, 2))

for i in range(n) :
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    
# plt.show() - 잘 나옴

# 컨볼루셔널 오토인코더 정의하기
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)
            ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size = 3, strides = 2, activation='relu',
                                   padding = 'same'),
            layers.Conv2DTranspose(16, kernel_size = 3, strides = 2, activation='relu',
                                   padding = 'same'),
            layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
            ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = Denoise()
autoencoder.compile(optimizer = 'adam', loss = losses.MeanSquaredError())
autoencoder.fit(x_train_noisy, x_train, epochs = 10, shuffle=True, 
                validation_data=(x_test_noisy, x_test))

# encoder 요약 살펴보기. 이미지가 28 x 28 에서 7 x 7 로 다운샘플링됨
autoencoder.encoder.summary()

# decoder 는 이미지를 7x7 에서 28 x 28 로 다시 업샘플링
autoencoder.decoder.summary()

# 오토인코더에서 생성된 노이즈가 있는 이미지와 노이즈가 제거된 이미지를 모두 플롯
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)

plt.show()

# 세번째 예 : 이상 감지
# ECG 데이터 로드
# Download the dataset
dataframe = pd.read_csv(
        'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head()) # print() added

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadrigoram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
        random_state=21)

# 데이터를 [0, 1]로 정규화
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# 데이터셋에서 1로 레이블이 지정된 정상 리듬만 사용하여 autoencoder 를 훈련
# 정상 리듬과 비정상 리듬을 분리
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[ train_labels ]
normal_test_data = test_data[ test_labels ]

anomalous_train_data = train_data[ ~train_labels ]
anomalous_test_data = test_data[ ~test_labels ]

# 정상적인 ECG 를 플롯
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()

# 비정상적인 ECG 를 플롯
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()

# 모델 빌드
class AnomalyDetector(Model) :
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

# 오토인코더는 일반 ECG 만 사용하여 훈련되지만, 전체 테스트셋을 사용하여 평가
history = autoencoder.fit(normal_train_data, normal_train_data, epochs=20, batch_size=512,
        validation_data=(test_data, test_data), shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show() # added

# 재구성 오류가 정상 훈련 예제에서 하나의 표준 편차보다 큰 경우, ECG 를 비정상으로 분류
# 훈련 세트의 정상 ECG, 오토인코더에 의해 인코딩 및 디코딩 된 후의 재구성, 재구성 오류 플롯
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 비정상적인 테스트 예제에서 비슷한 플롯 생성
encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_imgs[0], 'r')
plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# 이상 감지하기
# 훈련 세트에서 정상 ECG에 대한 재구성 오류를 플롯
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

# 평균보다 표준 편차가 높은 임계값을 선택
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

# 재구성 오류가 임계값보다 큰 경우 ECG 를 이상으로 분류
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, preds)))
  print("Precision = {}".format(precision_score(labels, preds)))
  print("Recall = {}".format(recall_score(labels, preds)))

preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)