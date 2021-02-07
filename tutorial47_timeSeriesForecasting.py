# https://www.tensorflow.org/tutorials/structured_data/time_series

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# 날씨 데이터셋
zip_path = tf.keras.utils.get_file(
        origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname = 'jena_climate_2009_2016.csv.zip',
        extract = True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

df.head() # need print()?

plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots = True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots = True)
# plt.show() # added

# 검사 및 정리하기
print(df.describe().transpose()) # print() added

# 풍속(wind velocity)
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame
print(df['wv (m/s)'].min()) # print() added

# 특성 엔지니어링
# 바람
plt.clf() # added
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins = (50, 50), vmax = 400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
#plt.show() # added

# 풍향과 속도 열을 바람 벡터로 변환하면 해석이 더 쉬움
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)') * np.pi / 180

# Calculate the wind x and y components
df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)

# 바람 벡터의 분포는 모델이 올바르게 해석하기에 훨씬 더 간단함
plt.clf() # added
plt.hist2d(df['Wx'], df['Wy'], bins = (50, 50), vmax = 400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
print(ax.axis('tight')) # print() added
#plt.show() #  added

# 시간
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24 * 60 * 60
year = (365.2425) * day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.clf() # added
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
#plt.show() # added

fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / (hours_per_year)

f_per_year = f_per_dataset / years_per_dataset
plt.clf() # added
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels = ['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
#plt.show() # added

# 데이터 분할 : 훈련 , 검증, 테스트를 70%, 20%, 10% 로 나눔. 분할하기 전에 셔플하지 않음
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

num_features = df.shape[1]

# 데이터 정규화 - 훈련 데이터만 사용하여 평균 및 표준 편차를 게산
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# 특성 분포 살펴보기
plt.clf() # added
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name = 'Column', value_name = 'Normalized')
plt.figure(figsize = (12, 6))
ax = sns.violinplot(x = 'Column', y = 'Normalized', data = df_std)
_ = ax.set_xticklabels(df.keys(), rotation = 90)
#plt.show()

# 데이터 창 작업
# 1. 인덱스 및 오프셋
class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df = train_df, val_df = val_df, 
            test_df = test_df, label_columns = None):
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None :
            self.label_columns_indices = {name: i  for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
                f'Total window size : {self.total_window_size}',
                f'Input indices : {self.input_indices}',
                f'Label indices : {self.label_indices}',
                f'Label column name(s) : {self.label_columns}'])

w1 = WindowGenerator(input_width = 24, label_width = 1, shift = 24, label_columns = ['T (degC)'])
print(w1) # print() added

w2 = WindowGenerator(input_width = 6, label_width = 1, shift = 1, label_columns = ['T (degC)'])
print(w2)

# 2. 분할
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis = -1)
    
    # Slicing doesn't preserve static shape information, so set the shapes manually.
    # This way the 'tf.data.Datasets' are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels

WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100 + w2.total_window_size]),
                           np.array(train_df[200:200 + w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are : (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape : {example_inputs.shape}')
print(f'Labels shape : {example_labels.shape}')

w2.example = example_inputs, example_labels
def plot(self, model = None, plot_col = 'T (degC)', max_subplots = 3):
    inputs, labels = self.example
    plt.figure(figsize = (12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label = 'Inputs', marker = '.', zorder = -10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        
        if label_col_index is None:
            continue
        
        plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors = 'k',
                label = 'labels', c = '#2ca02c', s = 64)
        
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker = 'X', edgecolor = 'k', label = 'Predictions', c = '#ff7f0e', s = 64)
        
        if n == 0:
            plt.legend()
    plt.xlabel('Time [h]')

WindowGenerator.plot = plot

plt.clf() # added
w2.plot()
#plt.show() # added

plt.clf() # added
w2.plot(plot_col = 'p (mbar)')
#plt.show() # added

# 4. tf.data.Dataset 만들기
# 시계열 DataFrame 을 가져와 preprocessing.timeseries_dataset_from_array 함수를 이용하여
# (input_window, label_window) 쌍의 tf.data.Dataset 으로 변환
def make_dataset(self, data):
    data = np.array(data, dtype = np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(data = data, targets = None,
            sequence_length = self.total_window_size, sequence_stride = 1, shuffle = True,
            batch_size = 32,)

    ds = ds.map(self.split_window)
    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    # Get and cache an example batch of 'inputs, labels' for plotting.
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the '.train' dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair
print(w2.train.element_spec) # print() added

for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

# 단일 스텝 모델 : 현재 조건에만 기초하여 미래로 1타임스텝 진행된 단일 특성 값을 예측
single_step_window = WindowGenerator(input_width = 1, label_width = 1, shift = 1,
        label_columns = ['T (degC)'])
print(single_step_window) # print() added 

# 기준
class Baseline(tf.keras.Model):
    def __init__(self, label_index = None):
        super().__init__()
        self.label_index = label_index
        
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

baseline = Baseline(label_index = column_indices['T (degC)'])
baseline.compile(loss = tf.losses.MeanSquaredError(), metrics = [tf.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose = 0)

wide_window = WindowGenerator(input_width = 24, label_width = 24, shift = 1,
        label_columns = ['T (degC)'])

print(wide_window) # print() added
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', baseline(single_step_window.example[0]).shape)

plt.clf() # added
wide_window.plot(baseline)
#plt.show() # added

# 선형 모델
linear = tf.keras.Sequential([tf.keras.layers.Dense(units = 1)])
print('Input shape:', single_step_window.example[0].shape)
print('output shape:', linear(single_step_window.example[0]).shape)

MAX_EPOCHS = 20
def compile_and_fit(model, window, patience = 2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience,
            mode = 'min')
    model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(),
            metrics = [tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs = MAX_EPOCHS, validation_data = window.val, 
            callbacks = [early_stopping])
    return history

history = compile_and_fit(linear, single_step_window)
val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose = 0)


plt.clf() # added
plt.bar(x = range(len(train_df.columns)), height = linear.layers[0].kernel[:, 0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_= axis.set_xticklabels(train_df.columns, rotation = 90)
#plt.show() # added

# 밀집
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dense(units = 1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose = 0)

CONV_WIDTH = 3
conv_window = WindowGenerator(input_width = CONV_WIDTH, label_width = 1, shift = 1, 
        label_columns = ['T (degC)'])

print(conv_window) # print() added

plt.clf() # added
conv_window.plot()
plt.title('Given 3h as input, predict 1h into the future.')
#plt.show() # added

# layer.Flatten 을 모델의 첫 번째 레이어로 추가하여 다중 입력 스텝 창에서 dense 모델을 훈련
multi_step_dense = tf.keras.Sequential([
    # Shape : (time, features) -> (time * features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 1),
    # Add back the time dimension.
    # Shape: (outputs) -> (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)
IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose = 0)

plt.clf() # added
conv_window.plot(multi_step_dense)
#plt.show() # added

print('Input shape:', wide_window.example[0].shape)
try:
    print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e :
    print(f'\n{type(e).__name__}:{e}')

# 컨볼루션 신경망
# layers.Flatten 과 첫 번째 layers.Dense 는 layers.Conv1D 로 대체
# 컨볼루션이 출력에서 시간 축을 유지하므로 layers.Reshape 는 필요 없음
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters = 32, kernel_size = (CONV_WIDTH,), activation = 'relu'),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 1),
])

print("Conv model on 'conv_window'")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)
IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose = 0)

print("Wide window")
print("Input shape:", wide_window.example[0].shape)
print("Labels shape:", wide_window.example[0].shape)
print("Output shape:", conv_model(wide_window.example[0]).shape)

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(input_width = INPUT_WIDTH, label_width = LABEL_WIDTH, shift = 1,
        label_columns = ['T (degC)'])

print(wide_conv_window) # print() added

plt.clf() # added
wide_conv_window.plot(conv_model)
#plt.show() # added

# 순환 신경망
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] -> [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences = True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units = 1)
])

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose = 0)

plt.clf() # added
wide_window.plot(lstm_model)
#plt.show() # added

# 성능 : 일반적으로 각 모델의 성능이 이전 모델보다 약간 더 좋음
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.clf() # added
plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label = 'Validation')
plt.bar(x + 0.17, test_mae, width, label = 'Test')
plt.xticks(ticks = x, labels = performance.keys(), rotation = 45)
_ = plt.legend()
#plt.show() # added

# 다중 출력 모델
single_step_window = WindowGenerator(
    # 'WindowGenerator' returns all features as labels if you don't set the 'label_columns' argument.
    input_width = 1, label_width = 1, shift = 1)
wide_window = WindowGenerator(input_width = 24, label_width = 24, shift = 1)

for example_inputs, example_labels in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

# 기준
baseline = Baseline()
baseline.compile(loss = tf.losses.MeanSquaredError(), metrics = [tf.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose = 0)

# 밀집
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dense(units = num_features)
])

history = compile_and_fit(dense, single_step_window)
IPython.display.clear_output()
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose = 0)

# RNN
# %%time # 이거 에러남. 무슨 의미인지.....구글링해도 안나오는 듯..
wide_window = WindowGenerator(input_width = 24, label_width = 24, shift = 1)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] -> [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences = True),
    # Shape -> [batch, time, features]
    tf.keras.layers.Dense(units = num_features)
])

history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose = 0)
print()

# 고급 : 잔여연결(Residual networks, Residual connections)
class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        
        # The prediction for each timestamp is the input from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta

#%%time
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences = True),
        tf.keras.layers.Dense(
            num_features,
            # The predicted deltas should start small. So initialize the output layer with zeros)
            kernel_initializer = tf.initializers.zeros)
    ]))

history = compile_and_fit(residual_lstm, wide_window)

IPython.display.clear_output()
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose = 0)
print()

# 성능 : 다중 출력 모델의 전반적인 성능
x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.clf() # added
plt.bar(x - 0.17, val_mae, width, label = 'Validation')
plt.bar(x + 0.17, test_mae, width, label = 'Test')
plt.xticks(ticks = x, labels = performance.keys(), rotation = 45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()
#plt.show()

for name, value in performance.items():
    print(f'{name:15s}: {value[1]:0.4f}')
    
# 다중 스텝 모델
OUT_STEPS = 24
multi_window = WindowGenerator(input_width = 24, label_width = OUT_STEPS, shift = OUT_STEPS)
plt.clf() # added
multi_window.plot()
#plt.show() # added
print(multi_window) # print() added

# 기준
class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])
    
last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss = tf.losses.MeanSquaredError(), 
        metrics = [tf.metrics.MeanAbsoluteError()])
multi_val_performance = {}
multi_performance = {}


multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.val, verbose = 0)
plt.clf() # added
multi_window.plot(last_baseline)
#plt.show() # added

class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs
    
repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss = tf.losses.MeanSquaredError(), 
        metrics = [tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose = 0)
plt.clf() # added
multi_window.plot(repeat_baseline)
#plt.show() # added

# 싱글샷 모델
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step
    # Shape [batch, time, features] ==> [batch, 1, features]
    tf.keras.layers.Lambda(lambda x : x[:, -1:, :]),
    # Shape ==> [batch, 1, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer = tf.initializers.zeros),
    # Shape ==> [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])    
])

history = compile_and_fit(multi_linear_model, multi_window)
IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose = 0)
plt.clf() # added
multi_window.plot(multi_linear_model)
#plt.show() # added

# 밀집
multi_dense_model = tf.keras.Sequential([
    # Take the last time step
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x : x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation = 'relu'),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer = tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose = 0)
plt.clf() # added
multi_window.plot(multi_dense_model)
#plt.show() # added

# CNN
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x : x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation = 'relu', kernel_size = (CONV_WIDTH)),
    # Shape => [batch, 1, out_steps * fetures]
    tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer = tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)
IPython.display.clear_output()
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose = 0)
plt.clf() # added
multi_window.plot(multi_conv_model)
#plt.show() # added

# RNN
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more 'lstm_units' just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences = False),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer = tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)
IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose = 0)
plt.clf() # added
multi_window.plot(multi_lstm_model)
plt.show() # added

# 고급 : 자기 회귀 모델
# RNN
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the 'warmup' method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state = True)
        self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units = 32, out_steps = OUT_STEPS)

def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => 9batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)
    
    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
print(prediction.shape) # print() added

def call(self, inputs, training = None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)
    
    # Insert the first prediction
    predictions.append(prediction)
    
    # Run the rest of the prediction steps
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states = state, training = training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)
        
    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

FeedBack.call = call

print('Output shape (batch, time, features)', feedback_model(multi_window.example[0]).shape)
history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose = 0)
multi_window.plot(feedback_model)

# 성능 : 모델 복잡성이증가함에 따라 이득이 분명하게 감소
x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.clf() # added
plt.bar(x - 0.17, val_mae, width, label = 'Validation')
plt.bar(x + 0.17, test_mae, width, label = 'Test')
plt.xticks(ticks = x, labels = multi_performance.keys(), rotation = 45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show() # added

for name, value in multi_performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')

