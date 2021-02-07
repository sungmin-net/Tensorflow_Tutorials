# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=ko

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# IMDB 데이터셋 다운로드

NUM_WORDS = 1000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words = NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension) 크기의 행렬 생성
    results  = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0 # results[i]의 특정 인덱스만 1로 설정
    return results

train_data = multi_hot_sequences(train_data, dimension = NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension = NUM_WORDS)

plt.plot(train_data[0])
plt.show() # 내가 추가함


# 기준 모델 생성
baseline_model = keras.Sequential([
    # .summary 메소드 때문에 input_shape 가 필요
    keras.layers.Dense(16, activation = 'relu', input_shape = (NUM_WORDS, )),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')    
])

baseline_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                       metrics = ['accuracy', 'binary_crossentropy'])
baseline_model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                16016     
_________________________________________________________________
dense_1 (Dense)              (None, 16)                272       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 17        
=================================================================
Total params: 16,305
Trainable params: 16,305
Non-trainable params: 0
_________________________________________________________________
'''

baseline_history = baseline_model.fit(train_data, train_labels, epochs = 20, batch_size = 512,
                                      validation_data = (test_data, test_labels), verbose = 2)

# 작은 모델 생성
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation = 'relu', input_shape = (NUM_WORDS,)),
    keras.layers.Dense(4, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

smaller_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['accuracy', 'binary_crossentropy'])
smaller_model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 4)                 4004      
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 20        
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 5         
=================================================================
Total params: 4,029
Trainable params: 4,029
Non-trainable params: 0
_________________________________________________________________
'''
smaller_history = smaller_model.fit(train_data, train_labels, epochs = 20, batch_size = 512,
                                    validation_data = (test_data, test_labels), verbose = 2)


# 큰 모델 생성
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation = 'relu', input_shape = (NUM_WORDS,)),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
]) 

bigger_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                     metrics = ['accuracy', 'binary_crossentropy'])
bigger_model.summary()
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 512)               512512    
_________________________________________________________________
dense_7 (Dense)              (None, 512)               262656    
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 513       
=================================================================
Total params: 775,681
Trainable params: 775,681
Non-trainable params: 0
_________________________________________________________________
'''
bigger_history = bigger_model.fit(train_data, train_labels, epochs = 20, batch_size = 512,
                                  validation_data = (test_data, test_labels), verbose = 2)

# 훈련 손실과 검증 손실 그래프 그리기
def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize = (16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--',
                       label = name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(),
                 label = name.title() + ' Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()    
    plt.xlim([0, max(history.epoch)])
    plt.show() # 내가 추가함
    
plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

# 과대적합을 방지하기 위한 전략
# 가중치를 규제하기
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
                       activation = 'relu', input_shape = (NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.001),
                       activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

l2_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                 metrics = ['accuracy', 'binary_crossentropy'])
l2_model_history = l2_model.fit(train_data, train_labels, epochs = 20, batch_size = 512,
                                validation_data = (test_data, test_labels), verbose = 2)
plot_history([('baseline', baseline_history), ('l2', l2_model_history)])

# 드롭아웃 추가하기
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation = 'relu', input_shape = (NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
]) 

dpt_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                  metrics = ['accuracy', 'binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels, epochs = 20, batch_size = 512,
                                  validation_data = (test_data, test_labels), verbose = 2)

plot_history([('baseline', baseline_history), ('dropout', dpt_model_history)])