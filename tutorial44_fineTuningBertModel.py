# https://www.tensorflow.org/official_models/fine_tuning_bert
# 아래 메시지와 함께 리소스에 접근되지 않음 - 클라우드의 TPU 이용 예제라서 skip 함.
'''
2021-02-01 18:53:12.629628: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Traceback (most recent call last):
  File "C:\eclipse\workspace\TensorflowTutorials\tutorial44_fineTuningBertModel.py", line 30, in <module>
    print(tf.io.gfile.listdir(gs_folder_bert)) # print() added
  File "C:\eclipse\Python-3.8.5\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 691, in list_directory_v2
    raise errors.NotFoundError(
tensorflow.python.framework.errors_impl.NotFoundError: Could not find directory gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

# 자원
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
print(tf.io.gfile.listdir(gs_folder_bert)) # print() added


# 텐서플로 허브에서 사전 학습된 BERT 인코더 얻기
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"


