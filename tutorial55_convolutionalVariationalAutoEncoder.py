# https://www.tensorflow.org/tutorials/generative/cvae
# 다음의 에러로 끝까지 동작을 안함.. 디버깅 필요.
'''
Traceback (most recent call last):
  File "tutorial55_convolutionalVariationalAutoEncoder.py", line 267, in <module>
    plot_latent_images(model, 20)
  File "tutorial55_convolutionalVariationalAutoEncoder.py", line 258, in plot_latent_images
    x_decoded = model.sample(z)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\def_function.py", line 862, in _call
    results = self._stateful_fn(*args, **kwds)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\function.py", line 2941, in __call__
    filtered_flat_args) = self._maybe_define_function(args, kwargs)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\function.py", line 3361, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\function.py", line 3196, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\framework\func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\def_function.py", line 634, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\eager\function.py", line 3887, in bound_method_wrapper
    return wrapped_fn(*args, **kwargs)
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\framework\func_graph.py", line 977, in wrapper
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:

    tutorial55_convolutionalVariationalAutoEncoder.py:129 sample  *
        return self.decode(eps, apply_sigmoid = True)
    tutorial55_convolutionalVariationalAutoEncoder.py:140 decode  *
        logits = self.decoder(z)
    C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\keras\engine\base_layer.py:998 __call__  **
        input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
    C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\keras\engine\input_spec.py:234 assert_input_compatibility
        raise ValueError('Input ' + str(input_index) + ' of layer ' +

    ValueError: Input 0 of layer sequential_1 is incompatible with the layer: : expected min_ndim=2, found ndim=1. Full shape received: (2,)

C:\eclipse\workspace\TensorflowTutorials>

'''

from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

# MNIST 데이터셋 로드

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

# tf.data 를 사용하여 데이터 배치 및 셔플 처리
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size)
        .batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size)
        .batch(batch_size))

# 인코더 및 디코더 네트워크 정의 : 두 개의 작은 ConvNet 을 사용
# 인코더 네트워크 : 근사 사후 분포 q(z|x)를 정의. 잠재 표현 z 의 조건부 분포를 지정하기 위한
# 매개변수 세트를 출력. 분포를 대각선 가우스로 모델링, 네트워크는 인수 분해된 가우스의 평균 및
# 로그-분산 매개변수를 출력. 수치 안정성을 위해 분산을 직접 출력하지 않고 로그-분산을 출력

# 디코더 네트워크 : 잠복 샘플 z 를 입력으로 사용하여 관측 값의 조건부 분포에 대한 매개변수를 출력
# 관측값 p(x|z)의 조건부 분포를 정의. 잠재 이전 분포 p(z)를 단위 가우스로 모델링

# 재매개변수화 트릭 : 역전파가 무작위 노드를 통해 흐를 수 없기 때문에 샘플링 작업에서 병목 현상
# 디코더 매개변수와 다른 매개변수 e 를 다음과 같이 사용하여 z 를 근사

# 네크워크 아키텍처 : 인코더는 두 개의 컨볼루셔널 레이어, 그리고 완전히 연결된 레이어 사용.
# 디코더 네트워크는 완전히 연결된 레이어와 그 뒤 세 개의 컨볼루션 전치 레이어(디컨볼루셔널 레이어)를
# 사용하여 아키텍처를 미러링. 미니 배치 사용으로 인한 추가 무질서도가 샘플링의 무질서에 더해져
# 불안정성을 높일 수 있으므르, VAE 훈련 시 배치 정규화를 사용하지 않는 것이 일반적

class CVAE(tf.keras.Model):
    # Convolutional variational autoencoder
    
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (28, 28, 1)),
                tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = (2, 2), 
                        activation = 'relu'),
                tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = (2, 2),
                        activation = 'relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
                tf.keras.layers.Dense(units = 7 * 7 * 32, activation = tf.nn.relu),
                tf.keras.layers.Reshape(target_shape = (7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2,
                        padding = 'same', activation = 'relu'),
                tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2,
                        padding = 'same', activation = 'relu'),
                # No activataion
                tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = 3, strides = 1,
                        padding = 'same'),
            ]
        )
    
    @tf.function
    def sample(self, eps = None):
        if eps is None:
            eps = tf.random.normal(shape = (100, self.latent_dim))
        return self.decode(eps, apply_sigmoid = True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape = mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid = False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

# 손실함수 및 옵티마이저 정의
optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis = 1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi), 
            axis = raxis)
    
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits = x_logit, labels = x)
    logpx_z = -tf.reduce_sum(cross_ent, axis = [1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    # Executes one training step and returns the loss.
    # This function computes the loss and gradients, and uses the latter to update the model's
    # parameters.
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
# 훈련하기
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) 
# so it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(shape = [num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize = (4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap = 'gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    
generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
    
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait = False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, 
            end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

plt.imshow(display_image(epoch))
# plt.axis('off') # Display images # 이 라인에서 에러 나서 막아봄

plt.show() # added

# 저장된 모든 이미지의 애니메이션 GIF 표시하기
anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode = 'I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

# 잠재공간에서숫자의 2D 형태 표시하기
def plot_latent_images(model, n, digit_size = 28):
    # plots n x n digit images decoded from the latent space.
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array(([xi, yi]))
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1)] = digit.numpy()
    
    plt.figure(figsize = (10, 10))
    plt.imshow(image, cmap = 'Greys_r')
    plt.axis('off')
    plt.show()

plot_latent_images(model, 20)

    
    

