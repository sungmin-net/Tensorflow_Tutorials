# https://www.tensorflow.org/tutorials/distribute/input

# 분산 데이터셋
import tensorflow as tf
import numpy as np
import os
print(tf.__version__)

global_batch_size = 16
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)

@tf.function
def train_step(inputs):
    features, labels = inputs
    return labels - 0.3 * features

# Iterate over the dataset using the for.. in construct
for inputs in dataset:
    print(train_step(inputs))

global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
# 1 global bach of data fed to the model in 1 step.
print(next(iter(dist_dataset)))

# 속성
# 일괄처리 - 배치 분산 예시 설명
# 샤드 - 설명이 어렵다..
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(64).batch(16)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)

# 프리페치 - 이것도 설명이 어렵다..
mirrored_strategy = tf.distribute.MirroredStrategy()
def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(64).batch(16)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2) # This prefetches 2 batches per device.
    return dataset

dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)

# 분산 반복자
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function
def train_step(inputs):
    features, labels = inputs
    return labels - 0.3 * features

for x in dist_dataset:
    # train_step trains the model using the dataset elements
    loss = mirrored_strategy.run(train_step, args = (x,))
    print("Loss is ", loss)

# iter 를 사용하여 명시적 반복자 생성
num_epochs = 10
steps_per_epoch = 5
for epoch in range(num_epochs):
    dist_iterator = iter(dist_dataset)
    for step in range(steps_per_epoch):
        # train_step trains the model using the dataset elements
        loss = mirrored_strategy.run(train_step, args = (next(dist_iterator),))
        # which is the same as
        # loss = mirrored_strategy.run(train_step, args = (dist_iterator.get_next(), ))
        print("Loss is ", loss)

@tf.function
def train_fn(iterator):
    for _ in tf.range(steps_per_loop):
        strategy.run(step_fn, args = (next(iterator),))

#  You can break the loop with get_next_as_optional by checking if the Optional contains value
global_batch_size = 4
steps_per_loop = 5
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "CPU:0"])
dataset = tf.data.Dataset.range(9).batch(global_batch_size)
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distribute_iterator):
    for _ in tf.range(steps_per_loop):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
            break;
        per_replica_results = strategy.run(lambda x:x, args = (optional_data.get_value(),))
        tf.print(strategy.experimental_local_results(per_replica_results))
train_fn(distributed_iterator)

# element_spec 속성 사용
global_batch_size = 16
epochs = 5
steps_per_epochs = 5
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function(input_signature = [dist_dataset.element_spec])
def train_step(per_replica_inputs):
    def step_fn(inputs):
        return 2 * inputs
    
    return mirrored_strategy.run(step_fn, args = (per_replica_inputs,))

for _ in range(epochs):
    iterator = iter(dist_dataset)
    for _ in range(steps_per_epoch):
        output = train_step(next(iterator))
        tf.print(output)

# 부분 패치..
'''
d = tf.data.Dataset.list_files(pattern, shuffle = False)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
d = d.interleave(tf.dta.TFRecordDataset, cycle_length = num_readers, block_length = 1)
d = d.map(parser_fn, num_parallel_calls = num_map_threads)
'''

mirrored_strategy = tf.distribute.MirroredStrategy()
dataset_size = 24
batch_size = 6
dataset = tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

def predict(index, inputs):
    outputs = 2 * inputs
    return index, outputs

result = {}
for index, inputs in dist_dataset:
    output_index, outputs = mirrored_strategy.run(predict, args = (index, inputs))
    indices = list(mirrored_strategy.experimental_local_results(output_index))
    rindices = []
    for a in indices:
        rindices.extend(a.numpy())
    outputs = list(mirrored_strategy.experimental_local_results(outputs))
    routputs = []
    for a in outputs:
        routputs.extend(a.numpy())
    for i, value in zip(rindices, routputs):
        result[i] = value
print(result)

# 임의의 텐서 입력에 experimental_distribute_values_from_function 을 사용하기
mirrored_strategy = tf.distribute.MirroredStrategy()
worker_devices = mirrored_strategy.extended.worker_devices

def value_fn(ctx):
    return tf.constant(1.0)

distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
for _ in range(4):
    result = mirrored_strategy.run(lambda x:x, args = (distributed_values, ))
    print(result)

# 생성기에서 입력한 경우 tf.data.Dataset.from_generator 사용하기
mirrored_strategy = tf.distribute.MirroredStrategy()
def input_gen():
    while True:
        yield np.random.rand(4)
        
# use Dataset.from_generator
dataset = tf.data.Dataset.from_generator(input_gen, output_types = (tf.float32),
        output_shapes = tf.TensorShape([4]))
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
iterator = iter(dist_dataset)
for _ in range(4):
    mirrored_strategy.run(lambda x:x, args = (next(iterator),))

