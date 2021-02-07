# https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

# Oxford-IIIT Pets 데이터셋 다운로드
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info = True)

# 이미지를 뒤집는 간단한 확징. 영상을 [0,1]로 정규화. 분할 마스크의 픽셀에 {1,2,3}이라는 레이블 부여
# 편의성을 위해 분할 마스크는 {0, 1, 2}로 변경
# class 0 : 애완동물이 속한 픽셀
# class 1 : 애완동물과 인접한 픽셀
# class 2 : 위에 속하지 않는 경우/주변 픽셀

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    
    input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask

# 데이터셋에는 이미 필요한 몫의 시험과 훈련이 포함되어 있으므로, 동일한 분할을 계속 사용
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# 이미지 예제와 데이터셋에서 대응하는 마스크 보기
def display(display_list):
    plt.figure(figsize = (15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

# 모델 정의
# 수정된 U-Net 을 사용. U-Net 은 인코더(다운샘플러)와 디코더(업샘플러)를 포함.
# 미리 훈련된 MobileNetV2 를 인코더로 사용 가능. 여기서는 미리 훈련된 MobileNetV2 의 중간 출력 사용

OUTPUT_CHANNELS = 3 # 픽셀당 3개의 가능한 라벨이 있음. 각 화소가 세 개의 class 로 다중 분류됨

# 인코더는 교육 과정 중 학습되지 않음
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)

# 이 층들의 활성화 이용
layer_names = [
    'block_1_expand_relu', # 64 x 64
    'block_3_expand_relu', # 32 x 32
    'block_6_expand_relu', # 16 x 16
    'block_13_expand_relu',# 8 x 8
    'block_16_project',    # 4 x 4    
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 특징 추출 모델을 만듦
down_stack = tf.keras.Model(inputs = base_model.input, outputs = layers)
down_stack.trainable = False

# 디코더(업샘플러)는 TensorFlow 예제에서 구현된 일련의 업샘플 블록
up_stack = [
    pix2pix.upsample(512, 3), # 4 x 4 -> 8 x 8
    pix2pix.upsample(256, 3), # 8 x 8 -> 16 x 16
    pix2pix.upsample(128, 3), # 16 x 16 -> 32 x 32
    pix2pix.upsample(64, 3),  # 32 x 32 -> 64 x 64
]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape = [128, 128, 3])
    x = inputs
    
    # 모델을 통해 다운샘플링
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # 건너뛰기 연결을 업샘플링으로 설정
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
        
    # 이 모델의 마지막 층
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides = 2, padding = 'same') 
            # 64 x 64 -> 128 x 128
            
    x = last(x)
    return tf.keras.Model(inputs = inputs, outputs = x) 

# 모델 훈련하기
# 네트워크가 멀티클래스 예측과 마찬가지로 픽셀마다 레이블을 할당하므로 sparse_categorical_crossentropy
# 손실 함수 사용. 각 채널은 클래스를 예측하는 방법을 배우려하여 권장되는 손실
# create_mask 함수는 네트워크의 출력을 사용하여 가장 높은 값을 가진 채널을 픽셀에 레이블로 할당
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer = 'adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy'])

# 만들어진 모델의 구조 살펴보기
# tf.keras.utils.plot_model(model, show_shapes = True) # 이거 띄우려면 graphviz 가 필요하고,
        # graphviz 는 brew install graphviz 로 설치하는 데 brew 는 윈도우에서 동작 안함
        # brew 는 google interview 의 그 brew 가 맞음

# 모델을 시험해보고 훈련 전에 예측한 것이 무엇인지 알아보기
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[... , tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else :
        display([sample_image, sample_mask, 
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

# 모델이 훈련하는 동안 어떻게 향상되는지 관찰하기 위한 콜백 함수
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait = True)
        show_predictions()
        print('\n에포크 이후 예측 예시 : {}\n'.format(epoch + 1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS # 이건 또 무슨 문법인가..

model_history = model.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH,
        validation_steps = VALIDATION_STEPS, validation_data = test_dataset,
        callbacks = [DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'bo', label = 'Validation_loss')
plt.title('Training and Validaion Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

# 예측하기
show_predictions(test_dataset, 3)





