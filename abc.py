from collections import OrderedDict

import tensorflow as tf
import numpy as np

np.set_printoptions(threshold='nan')

data = np.load('/fyp/data.npy')
data = data.tolist()

data_input = tf.placeholder(tf.float32, shape=[None, len(data[0])])

embed_params = OrderedDict([
    ('principal', {'pos': 0, 'max_index': 256, 'embed_dim': 4}),
    ('resource', {'pos': 2, 'max_index': 1024, 'embed_dim': 5}),
    ('request_ip', {'pos': 3, 'max_index': 1024, 'embed_dim': 5}),
    ('request_zone', {'pos': 4, 'max_index': 64, 'embed_dim': 3}),
    ('request_country', {'pos': 5, 'max_index': 256, 'embed_dim': 4}),
    ('resource_ip', {'pos': 6, 'max_index': 1024, 'embed_dim': 4}),
    ('resource_zone', {'pos': 7, 'max_index': 64, 'embed_dim': 3}),
    ('resource_country', {'pos': 8, 'max_index': 256, 'embed_dim': 4})
])

embeddings = OrderedDict()
inputs = OrderedDict()
for key in embed_params.keys():
    initial_embeddings = tf.Variable(tf.random_uniform([embed_params[key]['max_index'],
                                                        embed_params[key]['embed_dim']], -1.0, 1.0))
    inputs[key] = tf.cast(tf.gather(data_input, embed_params[key]['pos'], axis=-1), tf.int32)
    embeddings[key] = tf.nn.embedding_lookup(initial_embeddings, inputs[key])

embedded_inputs = tf.concat(embeddings.values(), 1)

operation = tf.one_hot(tf.cast(tf.gather(data_input, 1, axis=-1), tf.int32), 5)

cos_hour = tf.gather(data_input, [9], axis=-1)
sin_hour = tf.gather(data_input, [10], axis=-1)
cos_weekday = tf.gather(data_input, [11], axis=-1)
sin_weekday = tf.gather(data_input, [12], axis=-1)

processed_inputs = tf.concat([embedded_inputs, operation, cos_hour, sin_hour, cos_weekday, sin_weekday], 1)

# Parameters
learning_rate = 0.00005
training_epochs = 1000
batch_size = 32

enc_input = processed_inputs.get_shape().as_list()[1]
ae_layers = [20, 10, 2]

def encoder(x, hidden_nums):
    layers = [None]*len(hidden_nums)

    w0 = tf.Variable(tf.random_normal([x.get_shape().as_list()[1], hidden_nums[0]]))
    b0 = tf.Variable(tf.random_normal([hidden_nums[0]]))
    layers[0] = tf.nn.tanh(tf.add(tf.matmul(x, w0), b0))

    for i in xrange(1, len(layers)):
        w = tf.Variable(tf.random_normal([hidden_nums[i-1], hidden_nums[i]]))
        b = tf.Variable(tf.random_normal([hidden_nums[i]]))
        layers[i] = tf.nn.tanh(tf.add(tf.matmul(layers[i-1], w), b))

    return layers[-1]


def decoder(x, hidden_nums, ae_input):
    layers = [None]*len(hidden_nums)

    w0 = tf.Variable(tf.random_normal([hidden_nums[0], hidden_nums[1]]))
    b0 = tf.Variable(tf.random_normal([hidden_nums[1]]))
    layers[0] = tf.nn.tanh(tf.add(tf.matmul(x, w0), b0))

    for i in xrange(1, len(hidden_nums)-1):
        w = tf.Variable(tf.random_normal([hidden_nums[i], hidden_nums[i+1]]))
        b = tf.Variable(tf.random_normal([hidden_nums[i+1]]))
        layers[i] = tf.nn.tanh(tf.add(tf.matmul(layers[i-1], w), b))

    output_dim = ae_input.get_shape().as_list()[1]
    w_out = tf.Variable(tf.random_normal([hidden_nums[-1], output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))
    layers[-1] = tf.nn.tanh(tf.add(tf.matmul(layers[-2], w_out), b_out))
    
    return layers[-1]


# Construct model
encoded = encoder(processed_inputs, ae_layers)
decoded = decoder(encoded, ae_layers[::-1], processed_inputs)

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(decoded - processed_inputs, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = int(len(data)/batch_size)
    for i in xrange(training_epochs):
        for b in xrange(batches):
            batch_data = data[b*batch_size:(b+1)*batch_size-1]
            _, c =sess.run([optimizer, loss], feed_dict={data_input: batch_data})
        print 'Epoch:', i, 'Loss:' ,c
    print sess.run(processed_inputs, feed_dict={data_input: batch_data})[-1]
    print sess.run(decoded, feed_dict={data_input: batch_data})[-1]
