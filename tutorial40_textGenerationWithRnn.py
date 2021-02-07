# https://www.tensorflow.org/tutorials/text/text_generation
import tensorflow as tf
import numpy as np
import os
import time

# 셰익스피어 데이터셋 다운로드
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 데이터 읽기
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# 텍스트의 길이는 그 안에 있는 문자의 수
print("텍스트의 길이 : {}자".format(len(text)))

# 텍스트의 처음 250자 출력
print(text[:250])

# 파일의 고유 문자수를 출력
vocab = sorted(set(text))
print('고유 문자수 {}개'.format(len(vocab)))

# 텍스트 처리
# 고유 문자에서 인덱스로 매핑 생성
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char, _ in zip(char2idx, range(20)):
    print('    {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('    ...\n}')

# 텍스트에서 처음 13개의 문자가 숫자로 어떻게 매핑되었는지 보기
print('{} --- 문자들이 다음의 정수로 매핑되었습니다. ---> {}'.format(repr(text[:13]), 
        text_as_int[:13]))

# 단일 입력에 대해 원하는 문장의 최대 길이
seq_length = 100
examples_per_epoch = len(text) // seq_length # 이거는 주석이 아니고 다른 의미가 있는 거 같다.

# 훈련 샘플 / 타깃 만들기
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# 첫 번째 샘플의 타깃 값을 출력
for input_example, target_example in dataset.take(1):
    print("입력 데이터 : ", repr(''.join(idx2char[input_example.numpy()])))
    print("타깃 데이터 : ", repr(''.join(idx2char[target_example.numpy()])))
    
# 벡터의 각 인덱스는 하나의 타임 스탭으로 처리. 타임 스텝 0의 입력으로 모델은 'F' 의 인덱스를 받고,
# 다음 문자로 'i' 를 예측. 다음 타임 스탭에서도 반복하지만, 현재 입력 문자 외에 이전 타임 스텝의
# 컨텍스트를 고려

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("{:4d}단계".format(i))
    print("  입력: {}({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  예상 출력: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# 훈련 배치 생성
BATCH_SIZE = 64

# 데이터셋을 섞을 버퍼 크기(TF 데이터는 무한한 스퀀스와 함께 작동이 가능하도록 설계되었으며, 따라서
# 전체 스퀀스를 메모리에 섞지 않음. 대신 요소를 섞는 버퍼를 유지)
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)
print(dataset) # print() added

# 모델 설계
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True,
                recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)        
    ])
    return model
model = build_model(vocab_size = len(vocab), embedding_dim = embedding_dim, rnn_units = rnn_units,
        batch_size = BATCH_SIZE)

# 모델 사용
# 출력 형태 확인
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (배치 크기, 시퀀스 길이, 어휘 사전 크기)")

model.summary()

# 배열의 첫번째 샘플링 시도
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples = 1)
sampled_indices = tf.squeeze(sampled_indices, axis = -1).numpy()
print(sampled_indices) # print() added

# 훈련되지 않은 모델에 의해 예측된 텍스트를 보기 위해 복호화
print("입력: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("예측된 다음 문자: \n", repr("".join(idx2char[sampled_indices])))

# 모델 훈련
# 표준 sparse_softmax_crossentropy 손실 함수는 이전 차원의 예측과 교차 적용되어 이 문제에 적합
# 이 모델은 로짓을 반환하기 때문에 from_logits 플래그를 설정해야 함
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("예측 배열 크기(shape) : ", example_batch_predictions.shape, 
        "# (배치 크기, 시퀀스 길이, 어휘 사전 크기")
print("스칼라 손실 : ", example_batch_loss.numpy().mean())

model.compile(optimizer = 'adam', loss = loss)

# 체크포인트 구성
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix,
        save_weights_only = True)

# 훈련 실행
EPOCHS = 10
history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])

# 텍스트 생성
# 최근 체크포인트 복원
print(tf.train.latest_checkpoint(checkpoint_dir)) # print() added
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# 예측 루프
def generate_text(model, start_string):
    # 평가 단계 (학습된 모델을 사용하여 텍스트 생성)
    # 생성할 문자의 수
    num_generate = 1000
    # 시작 문자열을 숫자로 변환(벡터화)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    # 결과를 저장할 빈 문자열
    text_generated = []
    
    # 온도가 낮으면 더 예측 가능한 텍스트가 되고, 높으면 더 의외의 텍스트가 됨.
    # 최적의 셋팅을 찾기 위한 실험
    temperature = 1.0
    
    # 여기에서 배치 크기 == 1
    model.reset_states()
    for i in range(num_generate) : 
        predictions = model(input_eval)
        # 배치 차원 제거
        predictions = tf.squeeze(predictions, 0)
        # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()
        
        # 예측된 단어를 다음 입력으로 모델에 전달. 이전 은닉 상태와 함께.
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: ")) # 결과를 개선하는 방법은 더 오래 훈련 EPOCHS = 30

# 고급 : 맞춤식 훈련
# 훈련 루프를 해제하고 직접 구현. 먼저 RNN 상태 초기화 - 데이터셋(배치별로)을 반복하고 각각 연관된
# 예측을 계산 - 컨텍스트에서의 예측과 손실을 계산 - 모델 변수에 대한 손실의 기울기를 계산
# - 옵티마이저를 사용하여 이전 단계로 이동
model = build_model(vocab_size = len(vocab), embedding_dim = embedding_dim, rnn_units = rnn_units,
        batch_size = BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()

@ tf.function
def train_step(input, taret):
    with tf.GradientTape() as tape:
        predictions = model(input)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, predictions,
                from_logits = True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 훈련 횟수
EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()
    
    # 모든 에포크(Epoch)의 시작에서 은닉 상태를 초기화. 초기 은닉 상태는 None
    hidden = model.reset_states()
    for (batch_n, (input, target)) in enumerate(dataset):
        loss = train_step(input, target)
        
        if batch_n % 100 == 0:
            template = "에포크 {} 배치 {} 손실 {}"
            print(template.format(epoch + 1, batch_n, loss))
            
    # 모든 5 에포크 마다 체크포인트 모델 저장
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch = epoch))
        
    print('에포크 {} 손실 {:.4f}'.format(epoch + 1, loss))
    print('1 에포크 당 {} 초 소요\n'.format(time.time() - start))
model.save_weights(checkpoint_prefix.format(epoch = epoch))



         