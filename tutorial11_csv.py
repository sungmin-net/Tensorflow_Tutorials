# 210119_https://www.tensorflow.org/tutorials/load_data/csv?hl=ko

import functools
import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read
np.set_printoptions(precision = 3, suppress = True)

# data load
r'''
C:\Users\Sungmin\.keras\datasets>head train.csv
survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone
0,male,22.0,1,0,7.25,Third,unknown,Southampton,n
1,female,38.0,1,0,71.2833,First,C,Cherbourg,n
1,female,26.0,0,0,7.925,Third,unknown,Southampton,y
1,female,35.0,1,0,53.1,First,C,Southampton,n
0,male,28.0,0,0,8.4583,Third,unknown,Queenstown,y
0,male,2.0,3,1,21.075,Third,unknown,Southampton,n
1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n
1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n
1,female,4.0,1,1,16.7,Third,G,Southampton,n
'''

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size = 5, # Artificially small to make examples easier to show
        label_name = LABEL_COLUMN,
        na_value = "?",
        num_epochs = 1,
        ignore_errors = True,
        **kwargs)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
            
show_batch(raw_train_data)
print() # 추가

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck',
               'embark_town', 'alone']
temp_dataset = get_dataset(train_file_path, column_names = CSV_COLUMNS)
show_batch(temp_dataset)
print() # 추가

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']
temp_dataset = get_dataset(train_file_path, select_columns = SELECT_COLUMNS)
show_batch(temp_dataset)
print() # 추가

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, select_columns = SELECT_COLUMNS, 
        column_defaults = DEFAULTS)
show_batch(temp_dataset)
print() # 추가

# 모든 열을 묶는 간단한 함수
def pack(features, label):
    return tf.stack(list(features.values()), axis = -1), label 

packed_dataset = temp_dataset.map(pack)
for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())

print() # 추가    
show_batch(raw_train_data)

example_batch, labels_batch = next(iter(temp_dataset))

class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names
        
    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis = -1)
        features['numeric'] = numeric_features
        
        return features, labels

NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']

packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)
example_batch, labels_batch = next(iter(packed_train_data))

# 데이터 정규화
import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
print(desc)

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
    # center the data
    return (data-mean) / std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean = MEAN, std = STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn = normalizer, 
        shape = [len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
print(numeric_column) # print() 추가

print(example_batch['numeric']) # print 추가

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print(numeric_layer(example_batch).numpy()) # print 추가


CATEGORIES = {
    'sex' : ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']    
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key = feature, 
            vocabulary_list = vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    
# See what you just created
print(categorical_columns) # print() added

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])

# 결합된 전처리 레이어
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])

# 모델 빌드
model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(1),
        ])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer = 'adam',
        metrics = ['accuracy'])

# 훈련, 평가 및 예측하기
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
model.fit(train_data, epochs = 20)
test_loss, test_accuracy = model.evaluate(test_data)
print("\n\nTest Loss {}, Test Accuracy {}".format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted survival: {:.2%}".format(prediction[0]), " | Actual outcome: ",
            ("SURVIVED" if bool(survived) else "DIED"))
