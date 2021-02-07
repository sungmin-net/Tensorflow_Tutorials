# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os, json

# 입력 함수 - 입력데이터를 워커 인덱스로 샤딩하여 각 워커 프로세스가 데이터셋을 1/n 만큼 겹치지않게
# 나누어 가짐

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context = None):
    datasets, info = tfds.load(name = 'mnist', with_info = True, as_supervised = True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test'])
    
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    
    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                input_context.input_pipeline_id)
    
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 다중 워커 설정
os.environ['TF_CONFIG'] = json.dumps({
    'cluster' : {
        'worker' : ["localhost:12345", "localhost:23456"]
    },
    'task' : {'type': 'worker', 'index':0}
    
})

# 모델 정의하기
LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10),
    ])
    logits = model(features, training = False)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels = labels, predictions = predictions)
    
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, 
            reduction = tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL :
        return tf.estimator.EstimatorSpec(mode, loss = loss)
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, 
            train_op = optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()))
    
# MultiWorkerMirroredStrategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# 모델 훈련 및 평가하기
config = tf.estimator.RunConfig(train_distribute = strategy)
classifier = tf.estimator.Estimator(model_fn = model_fn, model_dir = '/tmp/multiworker', 
        config = config)
tf.estimator.train_and_evaluate(classifier, 
        train_spec = tf.estimator.TrainSpec(input_fn = input_fn),
        eval_spec = tf.estimator.EvalSpec(input_fn = input_fn)
)

# 다음 로그까지 진행하고 더 반응이 없음.. 멀티워커 쪽은 실행이 잘 안되는 듯 함.
'''
2021-01-26 14:41:36.183418: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2021-01-26 14:41:36.183870: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
WARNING:tensorflow:From C:\eclipse\workspace\TensorflowTutorials\tutorial29_multiWorkerTrainingWithEstimator.py:65: _CollectiveAllReduceStrategyExperimental.__init__ (from tensorflow.python.distribute.collective_all_reduce_strategy) is deprecated and will be removed in a future version.
Instructions for updating:
use distribute.MultiWorkerMirroredStrategy instead
2021-01-26 14:41:41.094806: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-01-26 14:41:41.096548: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2021-01-26 14:41:41.097008: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2021-01-26 14:41:41.104159: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: DESKTOP-JNIDLRD
2021-01-26 14:41:41.104723: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: DESKTOP-JNIDLRD
2021-01-26 14:41:41.105745: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-01-26 14:41:41.106610: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-01-26 14:41:41.107970: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-01-26 14:41:41.114806: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:301] Initialize GrpcChannelCache for job worker -> {0 -> localhost:12345, 1 -> localhost:23456}
2021-01-26 14:41:41.115644: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:411] Started server with target: grpc://localhost:12345
'''