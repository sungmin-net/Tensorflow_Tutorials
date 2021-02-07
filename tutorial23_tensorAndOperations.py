# https://www.tensorflow.org/tutorials/customization/basics

import tensorflow as tf

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# 연산자 오버로딩도 지원
print(tf.square(2) + tf.square(3))

# 각각의 텐서는크기와 데이터 타입을 가짐
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

import numpy as np

ndarray = np.ones([3, 3])

print("텐서플로 연산은 자동적으로 넘파이 배열을 텐서로 변환")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환")
print(np.add(tensor, 1))

print(".numpy() 메서드는 텐서를 넘파이 배열로 변환")
print(tensor.numpy())

# 왜 42, 43 인가?

# GPU 가속
x = tf.random.uniform([3, 3])
print("GPU 사용이 가능한가 : ")
print(tf.test.is_gpu_available())

print("텐서가 GPU#0 에 있는 가 : ")
print(x.device.endswith('GPU:0'))

# 명시적 장치 배치
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    
    result = time.time() - start
    print("10 loops : {:0.2f}ms".format(1000 * result))
    
# CPU 에서 강제 실행
print("On CPU : ")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)
    
# GPU#0가 이용 가능 시 GPU#0에서 강제 실행
if tf.test.is_gpu_available():
    print("On GPU : ")
    with tf.device("GPU:0"):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endwith("GPU:0")
        time_matmul(x)

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# csv 파일을 생성
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line1
    Line2
    Line3
    """)

ds_file = tf.data.TextLineDataset(filename)

# 변환 적용
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2) # 이게 뭐야?
ds_file = ds_file.batch(2)

# 반복
print('ds_tensors 요소 : ')
for x in ds_tensors:
    print(x)

print("\nds_file 요소: ")
for x in ds_file :
    print(x)
    

    
    