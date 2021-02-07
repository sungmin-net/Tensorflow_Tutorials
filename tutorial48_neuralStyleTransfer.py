# https://www.tensorflow.org/tutorials/generative/style_transfer

import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3 :
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# 이미지 다운로드 및 스타일 참조 이미지와 콘텐츠 이미지 선택
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# 입력 시각화
# 이미지를 불러오는 함수를 정의하고, 최대 이미지 크기를 512개 픽셀로 제한
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 이미지를 출력하기 위한 함수
def imshow(image, title = None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis = 0)
        
    plt.imshow(image)
    if title:
        plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
#plt.show() # added

# TF-Hub 를 통한 빠른 스타일 전이
import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')        
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
image = tensor_to_image(stylized_image) # 'image =' added
#image.show() # added

# 콘텐츠와 스타일 표현 정의하기
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top = True, weights = 'imagenet')
prediction_probabilities = vgg(x)
print(prediction_probabilities.shape) # print() added 

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
print([(class_name, prob) for (number, class_name, prob) in predicted_top_5]) # print() added

# 분류층을 제외한 VGG19 모델을 불러오고, 각 층의 이름 출력
vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
print()
for layer in vgg.layers:
    print(layer.name)

# 이미지의 스타일과 콘텐츠를 나타내기 위한 모델의 중간층들을 선택
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv2', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# 모델 만들기
def vgg_layers(layer_names):
    # 중간층의 출력 값을 배열로 반환하는 vgg 모델 생성
    # 이미지넷 데이터셋에 사전학습된 VGG 모델을 불러옴
    vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# 위 함수를 이용하여 모델 생성
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)

# 각 층의 출력에 대한 통계량을 살펴봄
for name, output in zip(style_layers, style_outputs):
    print(name)
    print(" 크기 : ", output.numpy().shape)
    print(" 최소값 : ", output.numpy().min())
    print(" 최대값 : ", output.numpy().max())
    print(" 평균 : ", output.numpy().mean())
    print()
    
# 스타일 계산하기 - 특성 벡터끼리의 외적을 구한 후 평균 값을 냄. tf.linalg.einsum 함수 이용
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)

# 스타일과 콘텐츠 추출하기
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, inputs):
        # [0, 1] 사이의 실수 값을 입력으로 받음
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name : value for content_name, value in zip(self.content_layers, 
                content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}

extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))

print('스타일:')
for name, output in sorted(results['style'].items()) :
    print("    ", name)
    print("    크기 : ", output.numpy().shape)
    print("    최소값 : ", output.numpy().min())
    print("    최대값 : ", output.numpy().max())
    print("    평균 : ", output.numpy().mean())
    print()
    
print('콘텐츠')
for name, output in sorted(results['content'].items()):
    print("    ", name)
    print("    크기 : ", output.numpy().shape)
    print("    최소값 : ", output.numpy().min())
    print("    최대값 : ", output.numpy().max())
    print("    평균 : ", output.numpy().mean())

# 경사하강법 실행
# 스타일과 콘텐츠의 타깃값 지정
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# 최적화시킬 변수를 정의하고 콘텐츠 이미지로 초기화
image = tf.Variable(content_image)

# 픽셀이 실수이므로 0과 1 사이로 클리핑하는 함수 정의
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)

# 옵티마이저 생성
opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)

# 최적화를 위해 전체 오차를 콘텐츠와 스타일 오차의 가중합으로 정의
style_weight = 1e-2
content_weight = 1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
            for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
            for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

# 구현한 알고리즘을 시험하기 위해 몇 단계 돌려 봄
train_step(image)
train_step(image)
train_step(image)
i = tensor_to_image(image) # 'i = ' added
i.show() # added

import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end = '')
    display.clear_output(wait = True)
    #display.display(tensor_to_image(image)) # need show()?
    i = tensor_to_image(image) # 'i = ' added
    i.show() # added
    print("훈련 스텝 : {}".format(step))

end = time.time()
print("전체 소요 시간 : {:.1f}".format(end - start))

# 총 변위 손실
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(content_image)
plt.clf() # added
plt.figure(figsize = (14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas : Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas : Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2 * y_deltas + 0.5), "Horizontal Deltas : Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2 * x_deltas + 0.5), "Vertical Deltas : Styled")
plt.show() # added

plt.clf()
plt.figure(figsize = (14, 10))
sobel = tf.image.sobel_edges(content_image)
plt.subplot(1, 2, 1)
imshow(clip_0_1(sobel[... , 0] / 4 + 0.5), "Horizontal Sobel-edges")
plt.subplot(1, 2, 2)
imshow(clip_0_1(sobel[... , 1] / 4 + 0.5), "Vertical Sobel-edges")

# 정규화의 오차는 각 값의 절대값의 합으로 표현
def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

print(total_variation_loss(image).numpy()) # print() added

# 내장 표준함수 이용
print(tf.image.total_variation(image).numpy()) # print() added

# 다시 최적화하기
# 가중치를 정의하고 train_step 함수에서 사용
total_variation_weight = 30
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

# 최적화할 변수를 다시 초기화
image = tf.Variable(content_image)

# 최적화 수행
import time
start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end = '')
    display.clear_output(wait = True)
    #display.display(tensor_to_image(image)) # blocked
    i = tensor_to_image(image) # added
    i.show()  # added
end = time.time()
print("전체 소요 시간 : {:.1f}".format(end - start))

# 마지막으로, 결과물을 저장
file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)

# 아래 코드는 코랩에서만 될 듯 함
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)
