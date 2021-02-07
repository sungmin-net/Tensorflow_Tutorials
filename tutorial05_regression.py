# https://www.tensorflow.org/tutorials/keras/regression?hl=ko

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# 데이터 구하기
dataset_path = keras.utils.get_file("auto-mpg.data", 
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path) # C:\Users\Sungmin\.keras\datasets\auto-mpg.data

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration',
                'Model Year', 'Origin' ]
raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment='\t',
                          sep = ' ', skipinitialspace = True)
dataset = raw_dataset.copy()
print(dataset.tail()) # print() 추가
'''
      MPG  Cylinders  Displacement  ...  Acceleration  Model Year  Origin
393  27.0          4         140.0  ...          15.6          82       1
394  44.0          4          97.0  ...          24.6          82       2
395  32.0          4         135.0  ...          11.6          82       1
396  28.0          4         120.0  ...          18.6          82       1
397  31.0          4         119.0  ...          19.4          82       1

[5 rows x 8 columns]
'''
# 데이터 정제하기
print(dataset.isna().sum())
'''
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
'''
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail()) # print() 추가
'''
      MPG  Cylinders  Displacement  Horsepower  ...  Model Year  USA  Europe  Japan
393  27.0          4         140.0        86.0  ...          82  1.0     0.0    0.0
394  44.0          4          97.0        52.0  ...          82  0.0     1.0    0.0
395  32.0          4         135.0        84.0  ...          82  1.0     0.0    0.0
396  28.0          4         120.0        79.0  ...          82  1.0     0.0    0.0
397  31.0          4         119.0        82.0  ...          82  1.0     0.0    0.0

[5 rows x 10 columns]
'''
# 데이터셋을 훈련셋과 테스트 셋으로 분할
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

# 데이터 조사
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind = 'kde')
plt.show() # 이 라인 추가함

train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
print(train_stats)
'''
              count         mean         std  ...     50%      75%     max
Cylinders     314.0     5.477707    1.699788  ...     4.0     8.00     8.0
Displacement  314.0   195.318471  104.331589  ...   151.0   265.75   455.0
Horsepower    314.0   104.869427   38.096214  ...    94.5   128.00   225.0
Weight        314.0  2990.251592  843.898596  ...  2822.5  3608.00  5140.0
Acceleration  314.0    15.559236    2.789230  ...    15.5    17.20    24.8
Model Year    314.0    75.898089    3.675642  ...    76.0    79.00    82.0
USA           314.0     0.624204    0.485101  ...     1.0     1.00     1.0
Europe        314.0     0.178344    0.383413  ...     0.0     0.00     1.0
Japan         314.0     0.197452    0.398712  ...     0.0     0.00     1.0

[9 rows x 8 columns]
'''

# 특성과 레이블 분리하기
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 데이터 정규화
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 모델 만들기
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation = 'relu', input_shape = [len(train_dataset.keys())]),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae', 'mse'])
    return model

model = build_model()
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                640       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 4,865
Trainable params: 4,865
Non-trainable params: 0
_________________________________________________________________
'''

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result) # print 추가
'''
[[-0.45148247]
 [-0.38858595]
 [-0.23332347]
 [-0.47284675]
 [-0.18987747]
 [-0.16592279]
 [-0.23129228]
 [-0.5520612 ]
 [-0.12931497]
 [-0.2730092 ]]
'''

# 모델 훈련
# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0 : print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(normed_train_data, train_labels, epochs = EPOCHS, validation_split = 0.2,
                    verbose = 0, callbacks = [PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail()) # print() 추가
'''
         loss       mae       mse   val_loss   val_mae    val_mse  epoch
995  2.475423  0.987996  2.475423  10.087530  2.411126  10.087530    995
996  2.266375  0.936650  2.266375  10.263149  2.411967  10.263149    996
997  2.508028  1.025554  2.508028   9.791262  2.411568   9.791262    997
998  2.532231  1.009268  2.532231   9.605910  2.359288   9.605910    998
999  2.462917  0.963094  2.462917   9.524253  2.349183   9.524253    999
'''

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure(figsize = (8, 12))
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
    
plot_history(history)

model = build_model()

# patience 매개변수는 성능 향상을 체크할 에포크 횟수
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
history = model.fit(normed_train_data, train_labels, epochs = EPOCHS, validation_split = 0.2,
                    verbose = 0, callbacks = [early_stop, PrintDot()])
plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 2)
print('테스트 셋의 평균 절대 오차 : {:5.2f} MPG'.format(mae))
# 3/3 - 0s - loss: 5.7580 - mae: 1.9005 - mse: 5.7580
# 테스트 셋의 평균 절대 오차 :  1.90 MPG

# 예측
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show() # 내가 추가함