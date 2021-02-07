# https://www.tensorflow.org/tutorials/generative/dcgan

import tensorflow as tf
print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

# 데이터셋 로딩 및 준비
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# 이미지를 [-1, 1] 로 정규화
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# 데이터 배치를 만들고 섞음
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 모델 만들기

# 생성자
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # 배치사이즈로 None 이 주어짐
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', 
            use_bias = False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same',
            use_bias = False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same',
            use_bias = False, activation = 'tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    
    return model

# 아직 훈련되지 않은 생성자를 이용해 이미지를 생성
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training = False)
plt.imshow(generated_image[0, :, :, 0], cmap = 'gray') # 나올까?
plt.show() # added


# 감별자
# 합성곱 신경망 기반의 이미지 분류기. 진짜는 양수, 가짜는 음수 값을 출력하도록 훈련
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', 
            input_shape = [28, 28, 1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# 손실함수와 옵티마이저 정의
# 이 메서드는 크로스 엔트로피 손실함수를 계산하기 위한 헬퍼함수를 반환
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

# 감별자 손실함수
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 생성자 손실함수 : 감별자를 얼마나 잘 속였는 지에 대해 수치화. 
# 잘 속였다면 감별자는 가짜이미지를 진짜(또는 1)로 분류함. 
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 감별자와 생성자가 따로 훈련되기 대문에, 감별자와 생성자의 옵티마이저가 다름
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 체크포인트 저장
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
        discriminator_optimizer = discriminator_optimizer, generator = generator,
        discriminator = discriminator)

# 훈련 루프 정의
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# 이 시드를 시간이 지나도 재활용(GIF 애니메이션에서 진전 내용을 시각화하기 쉬움)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 'tf.function'이 어떻게 사용되는 지 주목. 이 데코레이터는 함수를 컴파일함.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)
        
        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, 
            discriminator.trainable_variables))
    
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            train_step(image_batch)
            
        # GIF 를 위한 이미지를 바로 생성
        display.clear_output(wait = True)
        generate_and_save_images(generator, epoch + 1, seed)
        
        # 15 에포크가 지날 때마다 모델을 저장
        if (epoch + 1) % 15 == 0 :
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('에포크 {} 에서 걸린 시간은 {} 초 입니다.'.format(epoch + 1, time.time() - start))
        
    # 마지막 에포크가 끝난 후 생성
    display.clear_output(wait = True)
    generate_and_save_images(generator, epochs, seed)

# 이미지 생성 및 저장
def generate_and_save_images(model, epoch, test_input):
    # 'training' 이 False 로 맞춰진 것을 주목. 이렇게 하면 (배치 정규화를 모함하여) 모든 층들이
    # 추론 모드로 실행
    predictions = model(test_input, training = False)
    
    fig = plt.figure(figsize = (4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# 모델 훈련
train(train_dataset, EPOCHS)
print(checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))) # print() added

# GIF 생성
# 에포크 숫자를 사용하여 하나의 이미지를 보여줌
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
i = display_image(EPOCHS) # 'i = ' added
i.show() # added

# imageio 로 훈련 중에 저장된 이미지를 사용해 GIF 애니메이션을 만듦
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode ='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython
if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename = anim_file)

# 코랩 전용 코드
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(anim_file)

