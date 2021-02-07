# https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko

import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# 예제 데이터셋 받기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 간단한 Sequential 모델 정의
def  create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation = 'relu', input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    
    model.compile(optimizer = 'adam', 
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

# 모델 객체 생성
model = create_model()

# 모델 구조 출력
model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                 save_weights_only = True,
                                                 verbose = 1)

# 새로운 콜백으로 모델 훈련하기
model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels),
          callbacks = [cp_callback]) # 콜백을 훈련에 전달

# 옵티마이저의 상태를 저장하는 것과 관련한 경고가 발생할 수 있으나 무시해도 좋음

# 기본 모델 객체 생성  
model = create_model()

# 모델 평가
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print("훈련되지 않은 모델의 정확도 : {:5.2f}%".format(100 * acc))

# 가중치 로드
model.load_weights(checkpoint_path)

# 모델 재평가
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100 * acc))

# 체크포인트 콜백 매개변수 - 새로운 모델을 훈련하고 다섯 번의 에포크마다 체크포인트 저장
# 파일 이름에 에포크 번호를 포함 시킴('str.format' 포멧)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 다섯 번째 에포크마다 가중치를 저장하기 위한 콜백을 생성
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, verbose = 1,
                      save_weights_only = True, period = 5)

# 새로운 모델 객체를 생성
model = create_model()

# checkpoint_path 포멧을 사용하는 가중치를 저장
model.save_weights(checkpoint_path.format(epoch = 0))

# 새로운 콜백을 사용하여 모델을 훈련
model.fit(train_images, train_labels, epochs = 50, callbacks=[cp_callback],
        validation_data = (test_images, test_labels), verbose = 0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# 모델을 초기화하고 최근 체크포인트를 로드하여 테스트. 새로운 모델 생성
model = create_model()

# 이전에 저장한 가중치를 로드
model.load_weights(latest)

# 모델 재평가
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100*acc))


# 수동으로 가중치 저장
model.save_weights('./checkpoints/my_checkpoint')

# 새로운 모델 생성
model = create_model()

# 가중치를 복원
model.load_weights('./checkpoints/my_checkpoint')

# 모델을 평가
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100*acc))

# 전체 모델 저장하기 - model.save 메서드로 모델의 구조, 가중치, 훈련 설정을 하나의 파일/폴더에 저장
# SavedModel 포멧 - 모델을 직렬화 하는 다른 방법. tf.keras.models.load_model 로 복원

# 새로운 모델 객체를 만들고 훈련
model = create_model()
model.fit(train_images, train_labels, epochs = 5)

# SaveModel 로 전체 모델을 저장
model.save("saved_model/my_model")

# 저장된 모델로부터 새로운 케라스 모델을 로드
new_model = tf.keras.models.load_model('saved_model/my_model')

# 모델 구조 확인
new_model.summary()

# 복원된 모델 평가
loss, acc = new_model.evaluate(test_images, test_labels, verbose = 2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100 * acc))
print(new_model.predict(test_images).shape)


# HDF5 파일로 저장하기
model = create_model()
model.fit(train_images, train_labels, epochs = 5)

# 전체 모델을 HDF5 파일로 저장

# .h5 확장자는 이 모델이 HDF5로 저장되었다는 것을 나타냄.
model.save('my_model.h5')

# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성
new_model = tf.keras.models.load_model('my_model.h5')

# 모델 구조 출력
new_model.summary()

# 정확도 확인
loss, acc = new_model.evaluate(test_images, test_labels, verbose = 2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100 * acc))

# 사용자 정의 객체와 get_config 관련 예제는 아래 참고
# https://www.tensorflow.org/guide/keras/custom_layers_and_models?hl=ko