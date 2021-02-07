# 210119_https://www.tensorflow.org/tutorials/load_data/pandas_dataframe?hl=ko
# 데이터셋에 접근할 수 없음.

import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
# Exception: URL fetch failure on https://storage.googleapis.com/applied-dl/heart.csv: 404 -- Not Found

df = pd.read_csv(csv_file)
df.head()