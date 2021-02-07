# https://www.tensorflow.org/tutorials/generative/deepdream

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image

# 변환(dream-ify)할 이미지 선택하기
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# 이미지를 내려받아 넘파이 배열로 변환
def download(url, max_dim = None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin = url)
    img = PIL.Image.open(image_path)
    if max_dim : 
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

# 이미지 정규화
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

# 이미지 출력
def show(img):
    #display.display(PIL.Image.fromarray(np.array(img))) # blocked
    i = PIL.Image.fromarray(np.array(img)) # added
    i.show() # added
    
# 이미지 크기 조절
original_img = download(url, max_dim = 500)
show(original_img)
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
        # 출력 포기 -_ 


# 특성 추출 모델 준비
base_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')

# 선택한 층들의 활성화값을 최대화
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# 특성 추출 모델 생성
dream_model = tf.keras.Model(inputs = base_model.input, outputs = layers)

# 손실 계산하기 - 경사 하강법이 아닌 경사 상승법으로 손실을 최대화
def calc_loss(img, model):
    # 이미지를 순전파시켜 모델의 활성화 값을 얻음
    # 이미지의 배치(batch) 크기를 1로 만듦
    img_batch = tf.expand_dims(img, axis = 0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    
    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    
    return tf.reduce_sum(losses)

# 경사 상승법
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function(
        input_signature = (
            tf.TensorSpec(shape = [None, None, 3], dtype = tf.float32),
            tf.TensorSpec(shape = [], dtype = tf.int32),
            tf.TensorSpec(shape = [], dtype = tf.float32),)
    )
    
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # 'img'에 대한 그래디언트가 필요. 'GradientTape'은 기본적으로 오직 'tf.Variable'만 봄
                tape.watch(img)
                loss = calc_loss(img, self.model)
            
            # 입력 이미지의 각 픽셀에 대한 손실 함수의 그래디언트를 계산
            gradients = tape.gradient(loss, img)
            
            # 그래디언트를 정규화
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            
            # 경사상승법을 이용해 '손실'을 최대화하여 입력 이미지를 선택한 층들보다 더 '흥분'시킴
            # 그래디언트와 이미지의 차원이 동일하므로 그래디언트를 이미지에 직접 더해 이미지 업데이트
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            
        return loss, img

deepdream = DeepDream(dream_model)

# 주 루프
def run_deep_dream_simple(img, steps = 100, step_size = 0.01):
    # 이미지를 모델에 순전파하기 위해 uint8 형식으로 변환
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        
        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        
        display.clear_output(wait = True)
        show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))
        
    result = deprocess(img)
    display.clear_output(wait = True)
    show(result) # 이거 잘 나오는 듯.
    
    return result

dream_img = run_deep_dream_simple(img = original_img, steps = 100, step_size = 0.01)


# 한 옥타브 올라가기
import time
start = time.time()

OCTAVE_SCALE = 1.30
img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE**n), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    img = run_deep_dream_simple(img = img, steps = 50, step_size = 0.01)

display.clear_output(wait = True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img / 255.0, dtype = tf.uint8)
show(img)

end = time.time()
print(end - start) # print() added

# 선택사항 : 타일(tile)을 이용해 이미지 확장하기
def random_roll(img, maxroll):
    # 타일 경계선이 생기는 것을 방지하기 위해 이미지를 랜덤하게 이동
    shift = tf.random.uniform(shape = [2], minval = -maxroll, maxval = maxroll, dtype = tf.int32)
    shift_down, shift_right = shift[0], shift[1]
    img_rolled = tf.roll(tf.roll(img, shift_right, axis = 1), shift_down, axis = 0)
    return shift_down, shift_right, img_rolled

shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
show(img_rolled)

# 앞서 정의한 deepdream 함수에 타일 기능을 추가
class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function(
        input_signature = (
            tf.TensorSpec(shape = [None, None, 3], dtype = tf.float32),
            tf.TensorSpec(shape = [], dtype = tf.int32),)
    )
    
    def __call__(self, img, tile_size = 512):
        shift_down, shift_right, img_rolled = random_roll(img, tile_size)
        
        # 그래디언트를 0으로 초기화
        gradients = tf.zeros_like(img_rolled)
        
        # 타일이 하나만 있지 않은 이상 마지막 타일은 건너뜀
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])
        
        for x in xs:
            for y in ys:
                # 해당 타일의 그래디언트를 계산
                with tf.GradientTape() as tape:
                    # 'img_rolled' 에 대한 그래디언트를 계산
                    # 'GradientTape'은 기본적으로 오직 'tf.Variable'만 주시함
                    tape.watch(img_rolled)
                    
                    # 이미지에서 타일 하나를 추출
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = calc_loss(img_tile, self.model)
                
                # 해당 타일에 대한 이미지 그래디언트를 업데이트
                gradients = gradients + tape.gradient(loss, img_rolled)
                
        # 이미지와 그래디언트에 적용한 랜덤 이동을 취소
        gradients = tf.roll(tf.roll(gradients, -shift_right, axis = 1), -shift_down, axis = 0)
                            
        # 그래디언트를 정규화
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        
        return gradients

get_tiled_gradients = TiledGradients(dream_model)

def run_deep_dream_with_octaves(img, steps_per_octave =  100, step_size = 0.01, 
        octaves = range(-2, 3), octave_scale = 1.3):
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # 옥타브에 따라 이미지의 크기를 조정
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))
        
        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            
            if step % 10 == 0:
                display.clear_output(wait = True)
                show(deprocess(img))
                print('Octave {}, step {}'.format(octave, step))

    result = deprocess(img)
    return result

img = run_deep_dream_with_octaves(img = original_img, step_size = 0.01)

display.clear_output(wait = True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img / 255.0, dtype = tf.uint8)
show(img)