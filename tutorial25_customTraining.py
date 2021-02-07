# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
 
import os
import matplotlib.pyplot as plt

import tensorflow as tf

print("텐서플로 버전 : {}".format(tf.__version__))
print("즉시 실행 : {}".format(tf.executing_eagerly()))

# 붓꽃 분류 문제
# 훈련 데이터 가져오기 및 파싱

# 데이터셋 다운로드
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname = os.path.basename(train_dataset_url), 
        origin = train_dataset_url)
print("데이터셋이 복사된 위치 : {}".format(train_dataset_fp))

# 데이터 탐색 
# $head -n5 {train_dataset_fp} # 리눅스 명령어인듯 함

# CSV 파일 안에서 컬럼의 순서
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

print("특성: {}".format(feature_names))
print("레이블 : {}".format(label_name))

# 레이블 매핑 0: Iris setosa, 1: Iris versicolor, 2: Iris virginica

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, batch_size, 
        column_names = column_names, label_name = label_name, num_epochs = 1)

features, labels = next(iter(train_dataset))
print(features)

# 데이터 도식
plt.scatter(features['petal_length'], features['sepal_length'], c = labels, cmap = 'viridis')
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

# 텐서의 리스트로부터 값을 취해 특정 차원으로 결합된 텐서를 생성하는 tf.stack 메서드 사용
def pack_features_vector(features, labels):
    # 특성들을 단일 배열로 묶습니다.
    features = tf.stack(list(features.values()), axis = 1)
    return features, labels

# 각 features, label 쌍의 특성을 훈련 데이터 세트에 쌓기 위해 tf.data.Dataset.map 메서드 사용
train_dataset = train_dataset.map(pack_features_vector)

# cjt 5개 행의 샘플 살펴보기
features, labels = next(iter(train_dataset))
print(features[:5])

# 모델 타입 선정
# 케라스를 사용한 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = tf.nn.relu, input_shape = (4,)), # 입력의 형태가 필요
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(3)
])

predictions = model(features)
print(predictions[:5]) # print() added. 각 샘플은 각 클래스에 대한 로짓(logit)을 반환

# 로직을 각 클래스에 대한 확률로 변환하기 위해서 소프트 맥스 함수 사용
print(tf.nn.softmax(predictions[:5]))

print("  예측 : {}".format(tf.argmax(predictions, axis = 1)))
print("레이블 : {}".format(labels))

# 모델 훈련하기
# 손실 함수와 그래디언트 함수 정의하기
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true = y, y_pred = y_)

l = loss(model, features, labels)
print("손실 테스트 : {}".format(l))

# 모델을 최적화하기 위해 사용되는 그래디언트를 계산하기 위해 tf.GradientTape 컨텍스트 사용
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# 옵티마이저 생성
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

# 한번의 최적화 단계를 계산
loss_value, grads = grad(model, features, labels)
print("단계 : {}, 초기 손실 : {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("단계 : {},      손실 : {}".format(optimizer.iterations.numpy(), 
        loss(model, features, labels).numpy()))

# 훈련 루프
# 도식화를 위한 결과 저장
train_loss_results = []
train_accuracy_results = []

num_epochs = 201
for epoch in range(num_epochs) :
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # 훈련 루프 - 32개의 배치를 사용
    for x, y in train_dataset:
        # 모델 최적화
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # 진행 상황을 추적
        epoch_loss_avg(loss_value) # 현재 배치 손실을 추가
        # 예측된 레이블과 실제 레이블 비교
        epoch_accuracy(y, model(x))
        
    # epoch 종료
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    if epoch % 50 == 0:
        print("에포크 {:03d}: 손실 : {:.3f}, 정확도 : {:.3%}".format(epoch, 
                epoch_loss_avg.result(), epoch_accuracy.result()))

# 시간에 따른 손실함수 시각화
fig, axes = plt.subplots(2, sharex = True, figsize = (12, 8))
fig.suptitle('훈련 지표')

axes[0].set_ylabel("손실", fontsize = 14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("정확도", fontsize = 14)
axes[1].set_xlabel("에포크", fontsize = 14)
axes[1].plot(train_accuracy_results)
plt.show()

# 모델 유효성 평가
# 테스트 데이터셋 설정
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname = os.path.basename(test_url), origin = test_url)
test_dataset = tf.data.experimental.make_csv_dataset(test_fp, batch_size, 
        column_names = column_names, label_name = 'species', num_epochs = 1, shuffle = False)

test_dataset = test_dataset.map(pack_features_vector)

# 테스트 데이터셋을 사용한 모델 평가
# 훈련 단계와 다르게 테스트 데이터에 대해 오직 한번의 에포크 진행.
test_accuracy = tf.keras.metrics.Accuracy()
for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis = 1, output_type = tf.int32)
    test_accuracy(prediction, y)
    
print("테스트셋 정확도 : {:.3%}".format(test_accuracy.result()))
print(tf.stack([y, prediction], axis = 1)) # print() added

# 훈련된 모델로 예측하기
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)
for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("샘플 {} 예측 : {} ({:4.1f}%)".format(i, name, 100*p))

