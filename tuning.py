import os

from ocsvm import AEOneClassSVM

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.utils import shuffle

import matplotlib
import matplotlib.pyplot as plt
import itertools

tf.set_random_seed(2018)

LOG_DIR = os.path.join(os.getcwd(), 'logs')


gamma = 3.0


data = np.load('data.npy')[:, :16]
kf = KFold(n_splits=5, random_state=1)

data_input = tf.placeholder(tf.float32, shape=[None, 16])

with tf.variable_scope('embed'):
    embed_params = OrderedDict([
        ('resource', {'pos': 2, 'max_index': 8, 'embed_dim': 2}),
        ('request_ip', {'pos': 3, 'max_index': 66, 'embed_dim': 8})
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
    operation = tf.one_hot(tf.cast(tf.gather(data_input, 1, axis=-1), tf.int32), 6)

    request_country = tf.gather(data_input, list(range(4, 7)), axis=-1)
    resource_zone = tf.gather(data_input, list(range(7, 14)), axis=-1)

    time = tf.gather(data_input, list(range(14, 16)), axis=-1)
    processed_inputs = tf.concat([embedded_inputs, operation, request_country, resource_zone, time], 1)

nus = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for nu in nus:
    train_score = []
    test_score = []
    for train_index, test_index in kf.split(data):
        train, test = data[train_index], data[test_index]

        svm = AEOneClassSVM(processed_inputs, 1024, 'test', [20, 5], nu, 1.0, gamma, 100,
                            full_op=tf.train.AdamOptimizer(1e-3), svm_op=tf.train.AdamOptimizer(1e-4))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            svm.train_full(sess, train, input_tensor=data_input, dt=1e-7)
            svm.calibrate_svm(sess, train, input_tensor=data_input, dt=1e-7)

            sn = sess.run(svm.output, feed_dict={data_input: train})
            sc = sess.run(svm.output, feed_dict={data_input: test})
            train_score.append(1.0 * len([s for s in sn if s == 1]) / len(train))
            test_score.append(1.0*len([s for s in sc if s == 1])/len(test))

    print('Nu:', nu)
    print(train_score, test_score)
    print(np.mean(np.array(train_score)), np.mean(np.array(test_score)))

nu = 0.5
gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

for gamma in gammas:
    train_score = []
    test_score = []
    for train_index, test_index in kf.split(data):
        train, test = data[train_index], data[test_index]

        svm = AEOneClassSVM(processed_inputs, 1024, 'test', [20, 5], nu, 1.0, gamma, 100,
                            full_op=tf.train.AdamOptimizer(1e-3), svm_op=tf.train.AdamOptimizer(1e-4))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            svm.train_full(sess, train, input_tensor=data_input, dt=1e-7)
            svm.calibrate_svm(sess, train, input_tensor=data_input, dt=1e-7)

            sn = sess.run(svm.output, feed_dict={data_input: train})
            sc = sess.run(svm.output, feed_dict={data_input: test})
            train_score.append(1.0 * len([s for s in sn if s == 1]) / len(train))
            test_score.append(1.0*len([s for s in sc if s == 1])/len(test))

    print('Gamma:', gamma)
    print(train_score, test_score)
    print(np.mean(np.array(train_score)), np.mean(np.array(test_score)))