
# https://www.tensorflow.org/tutorials/keras/classification?hl=ko

# tensorflow 와 tf.keras 임포트
import tensorflow as tf
from tensorflow import keras

# 헬퍼 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # 2.3.0

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouse', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']

print(train_images.shape)   # (60000, 28, 28)
print(len(train_labels))    # 60000
print(train_labels)         # [9 0 0 ... 3 0 5]
print(test_images.shape)    # (10000, 28, 28)
print(len(test_labels))     # 10000

# 데이터 전처리
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[ train_labels[i] ])
plt.show()    


# 모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
    ])

# 모델 컴파일
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs = 5)

# 정확도 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

print("\n테스트 정확도 : ", test_acc)

predictions = model.predict(test_images)

print(predictions[0])
#[7.0671317e-06 1.3169701e-07 1.0449052e-06 1.4883516e-07 1.5105265e-06
# 8.5893102e-02 1.4153586e-05 9.0750314e-02 7.8387209e-05 8.2325411e-01]

print(np.argmax(predictions[0])) # 9

# 그래프 도식
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap = plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else :
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[ true_label ]),
                                         color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
# 0번째 원소의 이미지, 예측, 신뢰도 점수 배열 확인
i = 0
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# 12번째 원소의 이미지, 예측, 신뢰도 점수 배열 확인
i = 12
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# 처음 x개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력
# 올바른 예측은 파랑색으로, 잘못된 예측은 빨강색으로 출력
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# 훈련된 모델을 사용하여 하나의 이미지를 예측
img = test_images[0]
print(img.shape) # (28, 28)

# tf.keras 모델은 한번에 샘플의 묶음 도는 배치로 예측을 만드는 데 최적화되어 있으므로, 
# 하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 함
img = (np.expand_dims(img, 0))
print(img.shape) # (1, 28, 28)

# 이 이미지의 예측을 생성
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45) # 이 라인에서 그림이 하나 나와야 할 것 같은데..?
plt.show() # plt.show() 내가 추가함
print(np.argmax(predictions_single[0])) # print() 추가함