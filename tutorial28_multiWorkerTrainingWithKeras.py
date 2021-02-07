# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
# 210126

from __future__ import absolute_import, division, print_function, unicode_literals

'''
try:
#    %tensorflow_version 2.x    # %tensorflow_version 기능은 코렙에서만 사용 가능 
except Exception:
    pass
'''

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

# 데이터셋 준비하기

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# MNIST 데이터를 (0, 255] 범위에서 (0., 1] 범위로 조정
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

datasets, info = tfds.load(name = 'mnist', with_info = True, as_supervised = True)
train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

# 케라스 모델 만들기 : tf.keras.Sequential API 로 간단한 합성곱 신경망 케라스 모델을 만들고 컴파일
def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')        
    ])
    
    model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, 
            optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
            metrics = ['accuracy']                  
    )
    return model

# 단일 워커를 이용하여 적은 수의 에포크만큼 훈련
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(x = train_datasets, epochs = 3) 

# 다중 워커 구성
import os, json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster' : {
        'worker' : ["localhost:12345", "localhost:23456"]
    },
    'task' : {'type': 'worker', 'index':0}
    }
)

# 적절한 전략 고르기 - 각 훈련 단계가 워커들이 가진 복제본들끼리 동기화 되는 동기 훈련 방식 또는
# 동기화가 엄격하게 이루어지지 않는 비동기 훈련 방식
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# RuntimeError: Collective ops must be configured at program startup 여기부터 동작하지 않음

NUM_WORKERS = 2
# 여기에서 배치 크기는 워커의 수를 곱한 크기로 늘려야 한다. tf.data.Dataset.batch 에는 전역 배치
# 크기를 지정해야 하기 때문이다. 전에는 64였지만, 이제 128 이 된다.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x = train_datasets, epochs = 3)

# 다중 워커 훈련에서 수렴과 성을을 위해 데이터를 여러 부분으로 샤딩하는 데, tf.ditribute.Strategy 가
# 자동으로 수행한다. 훈련할 때 샤딩(sharding) 을 직접 하고 싶다면, 다음과 같이 자동 샤딩 기능을 끈다. 
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
train_datasets_no_auto_shard = train_datasets.with_options(options)

# 내결함성
# 동기 훈련 방식에서 워커 하나가 죽으면 전체 클러스타 죽고, 복구 메커니븜이 없음
# 모든 워커가 훈련 에포크 혹은 스텝에 따라 동기화되므로, 다른 워커들은 죽거나 정지당한 워커가 복구
# 될 때까지 대기

# ModelCheckpoint 콜백
# 다중 워커 훈련의 내결함 기능을 사용하려면 tf.keras.Model.fit() 호출 시에
# tf.keras.callbacks.ModelCheckpoint 인스턴스를 제공해야 함

# 'filepath' 매개변수를 모든 워커가 접근할 수 있는 경로로 설정
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = './keras-ckpt')]
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x = train_datasets, epochs = 3, callbacks = callbacks)