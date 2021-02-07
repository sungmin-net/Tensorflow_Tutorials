# https://www.tensorflow.org/tutorials/distribute/keras

# 필요한 패키지 가져오기
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os

# 데이터셋 다운로드
datasets, info = tfds.load(name = 'mnist', with_info = True, as_supervised = True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# 분산 전략 정의하기
strategy = tf.distribute.MirroredStrategy()
print("장치의 수 : {}".format(strategy.num_replicas_in_sync))

# 입력 파이프라인 구성하기
# 데이터셋 내 샘플의 수는 info.splits.total_num_examples 로도 얻을 수 있음
# 기본적으로 GPU 메모리에 맞추어 가능한 가장 큰 배치 크기를 사용. 이에 맞게 학습률도 조정
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# 픽셀의 값은 0 ~ 255 이므로, 0-1 범위로 정규화
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    
    return image, label

# 훈련 데이터의 순서를 섞고, 훈련을 위해 배치로 묶음
train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# 모델 만들기
with strategy.scope():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(10, activation = 'softmax')            
    ])
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(),
            metrics = ['accuracy'])
    
# 콜백 정의하기
# 체크포인트를 저장할 체크포인트 디렉터리를 지정합니다.
checkpoint_dir = './training_checkpoints'
# 체크포인트 파일의 이름
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# 학습률을 점점 줄이기 위한 함수. 필요한 함수를 직접 정의하여 사용할 수 있음
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# 에포크가 끝날 때마다 학습률을 출력하는 콜백
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        print("\n에포크 {}의 학습률은 {}입니다.".format(epoch + 1, model.optimizer.lr.numpy()))

callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir = './logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix, save_weights_only = True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
]

# 훈련과 평가
model.fit(train_dataset, epochs = 12, callbacks = callbacks)

# 모델의 성능 확인을 위해, 가장 최근 체크포이트를 불러온 후 테스트 데이터에 대해 evaluate 을 호출
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
eval_loss, eval_acc = model.evaluate(eval_dataset)
print("평가 손실 : {}, 평가 정확도 : {}".format(eval_loss, eval_acc))

# savedModel 로 내보내기
path = 'saved_model/'
#tf.keras.experimental.export_saved_model(model, path) # 이거 에러남

saved_model_path = "./my_model.h5"
model.save(saved_model_path)



# strategy.scope 없이 모델 불러오기
# unreplicated_model = tf.keras.experimental.load_from_saved_model(path) #이거 에러남
unreplicated_model = tf.keras.models.load_model(saved_model_path) 

unreplicated_model.compile(loss = 'sparse_categorical_crossentropy',
        optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print("평가 손실 : {}, 평가 정확도 : {}".format(eval_loss, eval_acc))

# strategy.scope 와 함께 모델 불러오기
with strategy.scope():    
    #replicated_model = tf.keras.experimental.load_from_saved_model(path) # 이거 에러남
    replicated_model = tf.keras.models.load_model(saved_model_path)
    replicated_model.compile(loss = 'sparse_categorical_crossentropy',
            optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print("평가 손실 : {}, 평가 정확도 : {}".format(eval_loss, eval_acc))

