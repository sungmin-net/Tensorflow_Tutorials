# https://www.tensorflow.org/tutorials/text/word_embeddings

# 텍스트를 숫자로 표현(벡터화)
# 원-핫 인코딩 설명

# 설정
import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# IMDB 데이터셋 다운로드
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar = True, cache_dir = '.',
        cache_subdir = '')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir)) # print() added

# train 디렉토리 살펴보기 : 긍정(pos) 및 부정(neg) 리뷰를 사용하여 이진 분류 모델 학습
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir)) # print() added

# 불필요한 추가 폴더 제거
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# train 디렉토리를 사용, 검증을 위해 20%로 분할된 훈련 및 검증 데이터 세트를 모두 생성
batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train',
        batch_size = batch_size, validation_split = 0.2, subset = 'training', seed = seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train',
        batch_size = batch_size, validation_split = 0.2, subset = 'validation', seed = seed)

# 데이터셋에서 몇 가지 영화 리뷰와 라벨 살펴보기
for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy(), text_batch.numpy()[i])
        
# 성능을 위한 데이터셋 구성
# I/O 가 차단되지 않도록 데이터를 로드할 때 사용하는 두 가지 방법 : .cache(), .prefetch()

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# Embedding 레이어 사용 : 정수 인덱스(특정 단어)를 고밀도 벡터(임베딩)로 매핑하는 조회 테이블
# 임베딩의 차원(또는 너비)은 Dense 레이어의 뉴런 수와 같이 문제에 잘 맞는지 확인하기 위해 실험하는 매개변수

# Embeding a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
print(result.numpy()) # print() added

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
print(result.shape) # print() added

# 텍스트 전처리
# Create a custom standardization function to strip HTML break tags '<br />'
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Vocabulary size and number of words in a sequence
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to integers.
# Note that the layer uses the custom standardization defined above
# Set maximum_sequence length as all samples are not of the same length
vectorize_layer = TextVectorization(
        standardize = custom_standardization, max_tokens = vocab_size, output_mode = 'int',
        output_sequence_length = sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y : x)
vectorize_layer.adapt(text_ds)

# 분류 모델 만들기 : 연속 단어 모음 스타일 모델
# TextVectorization 레이어는 문자열을 어휘 색인으로 변환. Embedding 레이어는 정수로 인코딩된 어휘를
# 사용하여 각 단어 인덱스에 대한 임베딩 벡터를 찾음. 결과 차원은 (batch, sequence, embedding)
# GlobalAveragePooling1D 계층은 시퀀스 차원을 평균화하여 각 예제에 대해 고정 길이 출력 벡터를 반환
# 고정길이 출력 벡터는 16개의 은닉 유닛이 있는 완전 연결(Dense) 계층을 통해 파이프됨
# 마지막 레이어는 단일 출력 노드와 조밀하게 연결

embedding_dim = 16
model = Sequential([vectorize_layer, Embedding(vocab_size, embedding_dim, name = 'embedding'),
        GlobalAveragePooling1D(), Dense(16, activation = 'relu'), Dense(1)])

# 모델 컴파일 및 훈련
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = 'logs')
model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics = ['accuracy'])
model.fit(train_ds, validation_data = val_ds, epochs = 15, callbacks = [tensorboard_callback])

# 모델 요약 살펴보기
model.summary()

# TensorBoard 에서 모델 측정 항목을 시각화 # 될까? > 안됨
'''
%load_ext tensorboard
%tensorboard --logdir logs
'''

# 훈련된 단어 임베딩을 검색하여 디스크에 저장
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# 디스크에 가중치를 기록.
out_v = io.open('vectors.tsv', 'w', encoding = 'utf-8')
out_m = io.open('metadata.tsv', 'w', encoding = 'utf-8')

for index, word in enumerate(vocab) :
    if index == 0 : continue # skip 0, it's padding
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')
out_v.close()
out_m.close()

# Colaboratory 에서 튜토리얼을 실행하는 경우 다음 코드로 파일을 로컬 머신에 다운로드
'''
try :
    from google.colab import files
    files.download('vectors.tsv')
    files.download('metadata.tsv')
except Exception as e:
    pass
'''
# 임베딩 시각화 - 이거는 노트북에서만 되는 듯..

