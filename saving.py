import os

from ocsvm import AEOneClassSVM

import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

tf.set_random_seed(2018)

LOG_DIR = os.getcwd()

nu = 0.1
gamma = 3.0

train = np.load('data.npy')[:, 1:16]

data_input = tf.placeholder(tf.float32, shape=[None, 15], name='data_input')

with tf.variable_scope('embed'):
    embed_params = OrderedDict([
        ('resource', {'pos': 1, 'max_index': 8, 'embed_dim': 2}),  # 8 is the number of resources
        ('request_ip', {'pos': 2, 'max_index': 66, 'embed_dim': 8})  # 66 is the number of IPs
    ])

    embeddings = OrderedDict()
    inputs = OrderedDict()
    for key in embed_params.keys():
        initial_embeddings = tf.Variable(tf.random_uniform([embed_params[key]['max_index'],
                                                            embed_params[key]['embed_dim']], -1.0, 1.0),
                                         name='{}_embedding'.format(key))
        inputs[key] = tf.cast(tf.gather(data_input, embed_params[key]['pos'], axis=-1), tf.int32)
        embeddings[key] = tf.nn.embedding_lookup(initial_embeddings, inputs[key])

    embedded_inputs = tf.concat(list(embeddings.values()), 1)
    operation = tf.one_hot(tf.cast(tf.gather(data_input, 0, axis=-1), tf.int32), 6)

    request_country = tf.gather(data_input, list(range(3, 6)), axis=-1)
    resource_zone = tf.gather(data_input, list(range(6, 13)), axis=-1)

    time = tf.gather(data_input, list(range(13, 15)), axis=-1)
    processed_inputs = tf.concat([embedded_inputs, operation, request_country, resource_zone, time], 1,
                                 name='processed_features')

    # processed_inputs size 28

svm = AEOneClassSVM(processed_inputs, 1024, 'test', [20, 10, 5], nu, 1.0, gamma, 100,
                    full_op=tf.train.AdamOptimizer(1e-3), svm_op=tf.train.AdamOptimizer(1e-4))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    svm.train_full(sess, train, input_tensor=data_input, dt=1e-7)
    svm.calibrate_svm(sess, train, input_tensor=data_input, dt=1e-7)
    train_processed = sess.run(processed_inputs, feed_dict={data_input: train})

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    config = projector.ProjectorConfig()
    for key in embed_params.keys():
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embed/{}_embedding'.format(key)
        embedding.metadata_path = '{}_metadata.tsv'.format(key)

    projector.visualize_embeddings(summary_writer, config)
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))

    np.save('processed_features.npy', train_processed)

summary_writer.close()
