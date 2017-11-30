import sys
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import yaml

settings_file = sys.argv[1]
with open(settings_file, 'r') as yaml_file:
    settings = yaml.load(yaml_file)

if settings.get('numpy_print_all', False):
    np.set_printoptions(threshold='nan')

data = np.load(settings['data_file'])
data = data.tolist()

data_input = tf.placeholder(tf.float32, shape=[None, len(data[0])])

embed_params = OrderedDict([
    ('principal',
     {'pos': 0,
      'max_index': settings.get('embeddings']['principal']['max'],
      'embed_dim': settings.get('embeddings']['principal']['embedded']}),
    ('resource',
     {'pos': 2,
      'max_index': settings.get('embeddings']['resource']['max'],
      'embed_dim': settings.get('embeddings']['resource']['embedded']}),
    ('request_ip',
     {'pos': 3,
      'max_index': settings.get('embeddings']['request_ip']['max'],
      'embed_dim': settings.get('embeddings']['request_ip']['embedded']}),
    ('request_zone',
     {'pos': 4, 
      'max_index': settings.get('embeddings']['request_zone']['max'],
      'embed_dim': settings.get('embeddings']['request_zone']['embedded']}),
    ('request_country',
     {'pos': 5, 
      'max_index': settings.get('embeddings']['request_country']['max'],
      'embed_dim': settings.get('embeddings']['request_country']['embedded']}),
    ('resource_ip',
     {'pos': 6, 
      'max_index': settings.get('embeddings']['resource_ip']['max'],
      'embed_dim': settings.get('embeddings']['resource_ip']['embedded']}),
    ('resource_zone',
     {'pos': 7, 
      'max_index': settings.get('embeddings']['resource_zone']['max'],
      'embed_dim': settings.get('embeddings']['resource_zone']['embedded']}),
    ('resource_country',
     {'pos': 8,
      'max_index': settings.get('embeddings']['resource_country']['max'],
      'embed_dim': settings.get('embeddings']['resource_country']['embedded']}),
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
learning_rate = settings['learning_rate']
training_epochs = settings['epochs']
batch_size = settings['batch_size']

enc_input = processed_inputs.get_shape().as_list()[1]
ae_layers = settings['autoencoder_layers']

def encoder(x, hidden_nums):
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

def ocsvm(x, nu, batch_size, ramp):
  features = x.get_shape().as_list()[1]
  w = tf.Variable(tf.truncated_normal([features, 1]))
  b = tf.Variable(0.0, tf.float32)
  
  rho = tf.Variable(0.0, tf.float32)
  ramped_rho = tf.multiply(ramp, rho)
  score = tf.add(tf.matmul(x, w), b)

  reg_loss = 0.5*tf.reduce_sum(tf.square(w))
  #hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros([batch_size, features]), rho - score)))
  ramp_loss = tf.cond(tf.reduce_sum(score - rho) >= 0,
                      lambda: tf.constant(0.0, tf.float32),
                      lambda: tf.cond(tf.reduce_sum(score - ramped_rho) <= 0,
                                      lambda: (rho - ramped_rho),
                                      lambda: tf.reduce_sum(rho - score))
                      )
                                      
  loss = tf.subtract(tf.add(reg_loss, tf.divide(tf.divide(ramp_loss, nu), batch_size)), rho)
  #loss = tf.subtract(tf.add(reg_loss, tf.divide(tf.divide(hinge_loss, nu), batch_size)), rho)
  output = tf.sign(score - rho)

  ww = tf.reduce_sum(tf.square(w))

  return output, loss

nu = settings['svm']['nu']
ramp = settings['svm']['ramp']
svm_class, svm_loss = ocsvm(processed_inputs, nu, batch_size, ramp)

ae_loss = tf.reduce_mean(tf.pow(decoded - processed_inputs, 2))
#total_loss = ac_loss + svm_loss
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

ae_op = tf.train.AdamOptimizer(learning_rate).minimize(ac_loss)
svm_op = tf.train.AdamOptimizer(learning_rate).minimize(svm_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    batches = int(len(data)/batch_size)
    print 'nu:', nu, 'initial rate:', learning_rate, 'batch size:', batch_size
    for i in xrange(1, training_epochs + 1):
        for b in xrange(batches):
            batch_data = data[b*batch_size:(b+1)*batch_size]
            ae_op_, ac_loss_ = sess.run([ae_op, ac_loss], feed_dict={data_input: batch_data})
            svm_op_, svm_loss_ = sess.run([svm_op, svm_loss], feed_dict={data_input: batch_data})

        if i % 50 == 0:
            #saver.save(sess,
            #           '/fyp/models/m-rate-{}-layers-{}-loss-{}'.format(
            #               learning_rate,
            #               '-'.join(str(x) for x in ae_layers),
            #               c
            #           )
            #)
            print 'Epoch:', i, 'Loss:'
            print 'svm:', svm_loss_
            print 'ae:', ae_loss_
            classes = sess.run(svm_class, feed_dict={data_input: data})
            print 100.0*len([c for c in classes if c[0] == 1.0])/len(classes), '%'
            print batch_data
            print sess.run(processed_inputs, feed_dict={data_input: batch_data})
            print sess.run(decoded, feed_dict={data_input: batch_data})
