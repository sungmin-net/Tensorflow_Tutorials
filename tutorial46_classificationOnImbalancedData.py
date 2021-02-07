# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
import tensorflow as tf
from tensorflow import keras

import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 데이터 처리 및 탐색
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
print(raw_df.head()) # print() added

print(raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()) 
        # print() added

neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total : {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

# 데이터 정리, 분할 및 정규화
cleaned_df = raw_df.copy()

# You don't want the 'Time' column.
cleaned_df.pop('Time')

# The 'Amount' column covers a huge range. Convert to log-space.
eps = 0.001 # 0 -> 0.1c
cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount')+eps)

# 데이터셋 분할
# Use a utility from sklearn to split and shuffle our dataset
train_df, test_df = train_test_split(cleaned_df, test_size = 0.2)
train_df, val_df = train_test_split(train_df, test_size = 0.2)

# Form up arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# sklearn.StandardScaler 를 사용하여 입력 기능을 정규화
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print("Training labels shape: ", train_labels.shape)
print("Validation labels shpae: ", val_labels.shape)
print("Test labels shape: ", test_labels.shape)

print("Training features shape: ", train_features.shape)
print("Validation features shape: ", val_features.shape)
print("Test features shape: ", test_features.shape)

# 데이터분포 살펴보기
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

sns.jointplot(pos_df['V5'], pos_df['V6'], kind = 'hex', xlim = (-5, 5), ylim = (-5, 5))
plt.suptitle("Positive distribution")

sns.jointplot(neg_df['V5'], neg_df['V6'], kind = 'hex', xlim = (-5, 5), ylim = (-5, 5))
_ =  plt.suptitle("Negative distribution")
#plt.show() # added

# 모델 및 메트릭 정의
METRICS = [
    keras.metrics.TruePositives(name = 'tp'),
    keras.metrics.FalsePositives(name = 'fp'),
    keras.metrics.TrueNegatives(name = 'tn'),
    keras.metrics.FalseNegatives(name ='fn'),
    keras.metrics.BinaryAccuracy(name = 'accuracy'),
    keras.metrics.Precision(name = 'precision'),
    keras.metrics.Recall(name = 'recall'),
    keras.metrics.AUC(name = 'auc'),
]

def make_model(metrics = METRICS, output_bias = None):
    if output_bias is not None :
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(16, activation = 'relu', input_shape = (train_features.shape[-1], )),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = output_bias),
    ])
    
    model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-3), loss = keras.losses.BinaryCrossentropy(),
            metrics = metrics)
    return model

# 기준 모델
# 모델 구축
EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_auc', verbose = 1, patience = 10,
        mode = 'max', restore_best_weights = True)
model = make_model()
model.summary()

print(model.predict(train_features[:10])) # print() added

# 선택사항 : 올바른 초기 바이어스 설정
results = model.evaluate(train_features, train_labels, batch_size = BATCH_SIZE, verbose = 0)
print("Loss: {:0.4f}".format(results[0])) # 단순 초기화

initial_bias = np.log([pos / neg])
print(initial_bias) # print() added

model = make_model(output_bias = initial_bias)
print(model.predict(train_features[:10])) # print() added
results = model.evaluate(train_features, train_labels, batch_size = BATCH_SIZE, verbose = 0)
print("Loss: {:0.4f}".format(results[0])) # 초기 가중치 계산 초기화(약 50배 적음)

# 초기 가중치 체크 포인트
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

# 바이어스 수정이 도움이 되는 지 확인 - 단순 초기화와 바이어스 수정의 20 epoch 모결 훈련 손실 비교
model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(train_features, train_labels, batch_size = BATCH_SIZE, epochs = 20,
        validation_data = (val_features, val_labels), verbose = 0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(train_features, train_labels, batch_size = BATCH_SIZE, epochs = 20,
        validation_data = (val_features, val_labels), verbose = 0)

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'], color = colors[n], 
            label = 'Train '+ label)
    plt.semilogy(history.epoch, history.history['val_loss'], color = colors[n], 
            label = 'Val' + label, linestyle = "--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')    
    plt.legend()

plt.clf() # added
plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
#plt.show() # added

# 모델 훈련
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(train_features, train_labels, batch_size = BATCH_SIZE, epochs = EPOCHS,
        callbacks = [early_stopping], validation_data = (val_features, val_labels))

# 학습 이력 확인
def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in  enumerate(metrics):
        name = metric.replace('_', ' ').capitalize()

        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color = colors[0], label = 'Train')
        plt.plot(history.epoch, history.history['val_' + metric], color = colors[0],
                linestyle = "--", label = 'Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])        
        plt.legend()
         
plt.clf() # added            
plot_metrics(baseline_history)
#plt.show() # added

# 메트릭 평가
train_predictions_baseline = model.predict(train_features, batch_size = BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size = BATCH_SIZE)

def plot_cm(labels, predictions, p = 0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize = (5, 5))
    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title('Confusion_matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions : ', np.sum(cm[1]))

baseline_results = model.evaluate(test_features, test_labels, batch_size = BATCH_SIZE, verbose = 0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ' : ', value)
print()

plt.clf() # added
plot_cm(test_labels, test_predictions_baseline)
#plt.show() # added

# ROC 플로팅
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    
    plt.plot(100 * fp, 100 * tp, label = name, linewidth = 2, **kwargs)
    plt.xlabel('False positivies [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

plt.clf() # added
plot_roc('Train Baseline', train_labels, train_predictions_baseline, color = colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color = colors[0],
        linestyle = '--')
plt.legend(loc = 'lower right')
#plt.show() # added

# 클래스 가중치
# Scaling by total / 2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0 : {:.2f}'.format(weight_for_0))
print('Weight for class 1 : {:.2f}'.format(weight_for_1))

# 클래스 가중치로 모델 교육
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(train_features, train_labels, batch_size = BATCH_SIZE,
        epochs = EPOCHS, callbacks = [early_stopping], validation_data = (val_features, val_labels),
        # The class weights go here
        class_weight = class_weight)

plt.clf() # added
plot_metrics(weighted_history)
#plt.show() # added

# 매트릭 평가
train_predictions_weighted = weighted_model.predict(train_features, batch_size = BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size = BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels, batch_size = BATCH_SIZE,
        verbose = 0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ' : ', value)
print()

plt.clf() # added
plot_cm(test_labels, test_predictions_weighted)
plt.show()

# 오버 샘플링
# 소수 계급 과대 표본 : 소스 클래스를 오버 샘플링하여 데이터셋을 리샘플링
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

# numpy 사용
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

print(res_pos_features.shape) # print() added

resampled_features = np.concatenate([res_pos_features, neg_features], axis = 0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis = 0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]

BUFFER_SIZE = 100000

def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)

for features, label in pos_ds.take(1):
    print("Features:\n", features.numpy())
    print()
    print("Label : ", label.numpy())

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights = [0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
    print(label.numpy().mean())

resampled_steps_per_epoch =np.ceil(2.0 * neg / BATCH_SIZE)
print(resampled_steps_per_epoch)

# 오버 샘플링 된 데이터에 대한 학습
resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

resampled_history = resampled_model.fit(resampled_ds, epochs = EPOCHS, 
        steps_per_epoch = resampled_steps_per_epoch, callbacks = [early_stopping],
        validation_data = val_ds)

plt.clf() # added
plot_metrics(resampled_history)
plt.show() # added

# 재교육 : 균형잡힌 데이터는 빠르게 학습되어 훈련 절차가 바르게 과적합 될 수 있으므로,
# epochs 를 분리하여 callbacks.EarlyStopping 을 제공
resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

resampled_history = resampled_model.fit(resampled_ds,
        # These are not real epochs
        steps_per_epoch = 20, epochs = 10 * EPOCHS, callbacks = [early_stopping],
        validation_data = (val_ds))

# 훈련 이력 재확인
plt.clf() # added
plot_metrics(resampled_history)
plt.show()

# 메트릭 평가
train_predictions_resampled = resampled_model.predict(train_features, batch_size = BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(test_features, batch_size = BATCH_SIZE)
resampled_results = resampled_model.evaluate(test_features, test_labels, batch_size = BATCH_SIZE,
        verbose = 0)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ' : ', value)
print

plt.clf()
plot_cm(test_labels, test_predictions_resampled)
plt.show()

# ROC 플로팅
plt.clf()
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color = colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color = colors[0], 
        linestyle = '--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color = colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color = colors[1],
        linestyle = '--')
                                       
plot_roc("Train Resampled", train_labels, train_predictions_resampled, color = colors[2])
plot_roc("Test Resampled", test_labels, test_predictions_resampled, color = colors[2],
        linestyle = '--')
plt.legend(loc = 'lower right')
plt.show()


