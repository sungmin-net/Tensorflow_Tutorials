# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=ko
# 210120 'TFReocrd 파일 작성'부터 에러나서 더 진행 불가

import tensorflow as tf
import numpy as np
import IPython.display as display

# The following functions can be used to convert a value to a type compatible with tf.train.Example.

def _bytes_feature(value):
    # Returns a byte_list from a string
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
    # Returns a float_list from a float / double
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
    # Returns an int64)list from a bool / enum / int / unit
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

feature = _float_feature(np.exp(1))
print(feature.SerializeToString()) # print() added

# The number of observations in the dataset
n_observations = int(1e4)

# Boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

feature3 = np.random.randn(n_observations)

def serialize_example(feature0, feature1, feature2, feature3):
    # creates a tf.train.Example message ready to be written to a file.
    # creates a dictionary mapping the feature name to the tf.train.Example-compatible data type
    feature = {
        'feature0' : _int64_feature(feature0),
        'feature1' : _int64_feature(feature1),
        'feature2' : _bytes_feature(feature2),
        'feature3' : _float_feature(feature3),
    }
    
    # create a features message using tf.train.Example
    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

# This is an example observation from the dataset.
example_observation = []
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print(serialized_example) # print() added

# 메시지 디코딩
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto) # print() added
    
print(tf.data.Dataset.from_tensor_slices(feature1))
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print(features_dataset) # print() added    

# Use 'take(1)' to only pull one example from the dataset.
for f0, f1, f2, f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)
    
def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
            serialize_example,
            (f0, f1, f2, f3),  # pass there args to the above function
            tf.string)         # the return type is 'tf.string'
    return tf.reshape(tf_string, ()) # The result is a scalar
        
print(tf_serialize_example(f0, f1, f2, f3)) # print() added # 에러남.. 더 진행 불가
