# https://www.tensorflow.org/tutorials/distribute/save_and_load

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
    datasets, ds_info = tfds.load(name = 'mnist', with_info = True, as_supervised = True)
    mnist_train, mnist_test = datasets['train'], datasets['test']
    
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
    
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    
    return train_dataset, eval_dataset

def get_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape = (28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
        
        return model

# 모델 훈련 시키기
model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs = 2)

# 모델 저장하고 불러오기
'''
고수준 케라스 model.save와 tf.keras.models.load_model
저수준 케라스 tf.saved_model.save와 tf.saved_model.load
'''

keras_model_path = "/tmp/keras_save"
model.save(keras_model_path) # save()는 전략 범위를 벗어나 호출되어야 함

# tf.distribute.Strategy 없이 모델 복원시키기
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs = 2)

another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
with another_strategy.scope():
    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
    restored_keras_model_ds.fit(train_dataset, epochs = 2)
    
# tf.saved_model 형 API 
model = get_model() # 새 모델 얻기
saved_model_path = '/tmp/tf_save'
tf.saved_model.save(model, saved_model_path)

DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

predict_dataset = eval_dataset.map(lambda image, label : image)
for batch in predict_dataset.take(1):
    print(inference_func(batch))
    
# 분산 방식으로 불러오고 추론할 수있음
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
    
    dist_predict_dataset = another_strategy.experimental_distribute_dataset(predict_dataset)
    
    # 분산 방식으로 기능 호출하기
    for batch in dist_predict_dataset:
        another_strategy.run(inference_func, args = (batch,))

# 불러온 객체를 케라스 층에 싸서(wrap) 더 큰 모델에 내장
import tensorflow_hub as hub
def build_model(loaded):
    x = tf.keras.layers.Input(shape = (28, 28, 1), name = 'input_x')
    # kerasLayer 로 감싸기
    keras_layer = hub.KerasLayer(loaded, trainable = True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model

another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    model = build_model(loaded)
    
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
    model.fit(train_dataset, epochs = 2)

# 어떤 api 를 사용해야 할까요? 케라스 모델을 가져올 수 없으면 tf.saved_model.load() 를 사용
# 그 외에는 tf.keras.models.load_model() 을 사용. 케라스 모델을 저장한 경우에만 케라스 모델 반환

model = get_model()
# 케라스의 save() API를 사용하여 모델 저장
model.save(keras_model_path)

another_strategy = tf.distribute.MirroredStrategy()
# 저수준 API를 사용하여 모델 불러오기
with another_strategy.scope():
    loaded = tf.saved_model.load(keras_model_path)
    
# 주의 사항
# 잘 정의되지 않은 입력을 가진 케라스 모델, 예를 들어 순차 모델은 입력 형태 없이 생성 가능
# 하위 분류된 모델들 또한 초기화 후에 잘 정의된 입력을 가지고 있지 않음. 이 경우 모델을 저장하고
# 불러올 시에 저수준 API 를 사용해야 하며, 아닌 경우 오류 발생 가능. 모델이 잘 정의된 입력을 가지고
# 있는 지 확인하려면, model.inputs 가 None 인지 확인. None 이 아니라면 잘 정의된 입력.
# 입력 형태는 모델이 .fit, .evaluate, .predict 에서 쓰이거나 모델을 호출(model(inputs)할 때
# 자동으로 정의됨. 아래는 예시.
class SubClassedModel(tf.keras.Model):
    output_name = 'output_layer'
    def __init__(self):
        super(SubClassedModel, self).__init__()
        self._dense_layer = tf.keras.layers.Dense(5, dtype = tf.dtypes.float32, 
                name = self.output_name)
    
    def call(self, inputs):
        return self._dense_layer(inputs)

my_model = SubClassedModel()
# my_model.save(keras_model_path) # 오류
tf.saved_model.save(my_model, saved_model_path)

