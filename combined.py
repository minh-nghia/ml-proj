from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.kernel_methods.python.mappers.random_fourier_features import RandomFourierFeatureMapper as RFFM
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

data = np.load('data.npy')
data = data.tolist()

data_input = tf.placeholder(tf.float32, shape=[None, len(data[0])])

with tf.variable_scope('embed'):
    embed_params = OrderedDict([
        ('principal', {'pos': 0, 'max_index': 256, 'embed_dim': 2}),
        ('resource', {'pos': 2, 'max_index': 1024, 'embed_dim': 3}),
        ('request_ip', {'pos': 3, 'max_index': 1024, 'embed_dim': 3}),
        ('request_zone', {'pos': 4, 'max_index': 64, 'embed_dim': 1}),
        ('request_country', {'pos': 5, 'max_index': 256, 'embed_dim': 2}),
        ('resource_ip', {'pos': 6, 'max_index': 1024, 'embed_dim': 3}),
        ('resource_zone', {'pos': 7, 'max_index': 64, 'embed_dim': 1}),
        ('resource_country', {'pos': 8, 'max_index': 256, 'embed_dim': 2})
    ])

    embeddings = OrderedDict()
    inputs = OrderedDict()
    for key in embed_params.keys():
        initial_embeddings = tf.Variable(tf.random_uniform([embed_params[key]['max_index'],
                                                            embed_params[key]['embed_dim']], -1.0, 1.0))
        inputs[key] = tf.cast(tf.gather(data_input, embed_params[key]['pos'], axis=-1), tf.int32)
        embeddings[key] = tf.nn.embedding_lookup(initial_embeddings, inputs[key])

    embedded_inputs = tf.concat(embeddings.values(), 1)

    operation = tf.one_hot(tf.cast(tf.gather(data_input, 1, axis=-1), tf.int32), 3)

    cos_hour = tf.gather(data_input, [9], axis=-1)
    sin_hour = tf.gather(data_input, [10], axis=-1)
    cos_weekday = tf.gather(data_input, [11], axis=-1)
    sin_weekday = tf.gather(data_input, [12], axis=-1)

    processed_inputs = tf.concat([embedded_inputs, operation, cos_hour, sin_hour, cos_weekday, sin_weekday], 1)

# Parameters
learning_rate = 1e-3
training_epochs = 3000
batch_size = 32
alpha = 1e-4
nu = 0.1
rbf_kernel = True
kernel_approx_features = 1000

enc_input = processed_inputs.get_shape().as_list()[1]
ae_layers = [20, 10, 2]

def encoder(x, hidden_nums):
    with tf.variable_scope('autoencoder'):
        layers = [None]*len(hidden_nums)

        w0 = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], hidden_nums[0]]))
        b0 = tf.Variable(tf.truncated_normal([hidden_nums[0]]))
        layers[0] = tf.nn.tanh(tf.add(tf.matmul(x, w0), b0))

        for i in xrange(1, len(layers)):
            w = tf.Variable(tf.truncated_normal([hidden_nums[i-1], hidden_nums[i]]))
            b = tf.Variable(tf.truncated_normal([hidden_nums[i]]))
            layers[i] = tf.nn.tanh(tf.add(tf.matmul(layers[i-1], w), b))

    return layers[-1]


def decoder(x, hidden_nums, ae_input):
    with tf.variable_scope('autoencoder'):
        layers = [None]*len(hidden_nums)

        w0 = tf.Variable(tf.truncated_normal([hidden_nums[0], hidden_nums[1]]))
        b0 = tf.Variable(tf.truncated_normal([hidden_nums[1]]))
        layers[0] = tf.nn.tanh(tf.add(tf.matmul(x, w0), b0))

        for i in xrange(1, len(hidden_nums)-1):
            w = tf.Variable(tf.truncated_normal([hidden_nums[i], hidden_nums[i+1]]))
            b = tf.Variable(tf.truncated_normal([hidden_nums[i+1]]))
            layers[i] = tf.nn.tanh(tf.add(tf.matmul(layers[i-1], w), b))

        output_dim = ae_input.get_shape().as_list()[1]
        w_out = tf.Variable(tf.truncated_normal([hidden_nums[-1], output_dim]))
        b_out = tf.Variable(tf.truncated_normal([output_dim]))
        layers[-1] = tf.nn.tanh(tf.add(tf.matmul(layers[-2], w_out), b_out))

        return layers[-1]

encoded = encoder(processed_inputs, ae_layers)
decoded = decoder(encoded, ae_layers[::-1], processed_inputs)

def ocsvm(x_in, nu, input_size, rbf_kernel=False, kernel_approx_features=1000):
    with tf.variable_scope('svm'):
        if rbf_kernel:
            features_in = x_in.get_shape().as_list()[1]
            rffm = RFFM(features_in, kernel_approx_features, seed=2018)
            x = rffm.map(x_in)
        else:
            x = x_in
        features = x.get_shape().as_list()[1]
        w = tf.Variable(tf.truncated_normal([features, 1]))
        rho = tf.Variable(0.0, tf.float32)        

        margin = rho - tf.matmul(x, w)
        reg_loss = nu*tf.reduce_sum(tf.square(w))
        hinge_loss = tf.reduce_sum(margin + tf.abs(margin))

        loss = reg_loss - 2*rho*nu + tf.divide(hinge_loss, input_size)
        output = tf.sign(-margin)

        return output, loss, w, rho

test_input = tf.placeholder(tf.float32, shape=[None, len(data[0])])

svm_class, svm_loss, svm_w, svm_b = ocsvm(encoded, nu, batch_size,
                                          rbf_kernel=rbf_kernel,
                                          kernel_approx_features=kernel_approx_features)
ae_loss = tf.reduce_mean(tf.pow(decoded - processed_inputs, 2))

loss = ae_loss + alpha*svm_loss
op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    batches = int(len(data)/batch_size)
    print nu, 'initial rate:', learning_rate, 'batch size:', batch_size
    for i in xrange(1, training_epochs + 1):
        for b in xrange(batches):
            batch_data = data[b*batch_size:(b+1)*batch_size]
            op_, loss_ = sess.run([op, loss], feed_dict={data_input: batch_data})
        
        if i % 100 == 0:
            print 'Epoch:', i, 'Loss:', loss_
            classes = sess.run(svm_class, feed_dict={data_input: data})
            print len([c for c in classes if c[0] == -1.0]), len(data)

    xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))

    Z = sess.run(svm_class, feed_dict={encoded: np.c_[xx.ravel(), yy.ravel()]})
    Z = Z.reshape(xx.shape)

    enc = sess.run(encoded, feed_dict={data_input: data})

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.scatter([d[0] for d in enc], [d[1] for d in enc], c='gold', edgecolors='k')
plt.show()
