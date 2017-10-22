import tensorflow as tf

inputt = [1, 4, 2, 3, 6, 7, 2, 4, 9, 10, 4, 11, 2, 4, 7, 2, 4, 3, 12, 10, 4]*10
inputs = inputt*1000

tests = [4, 5]

# Parameters
learning_rate = 0.01
training_epochs = 6
batch_size = 256
 
# Network Parameters
n_hidden_1 = 32
n_hidden_2 = 16
n_hidden_3 = 8
n_input = 64

user_size = 2048
embedded_user_size = 64
user_embeddings = tf.Variable(tf.random_uniform([user_size, embedded_user_size], -1.0, 1.0), trainable=False)
init = tf.global_variables_initializer()
with tf.Session() as random_sess:
    random_sess.run(init)
    user_embeddings = tf.constant(random_sess.run(user_embeddings))

user_inputs = tf.placeholder(tf.int32, shape=[None])
embedded_user = tf.nn.embedding_lookup(user_embeddings, user_inputs)
    
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}
 
def encoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    
    return layer_3

def decoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    
    return layer_3
 
# Construct model
encoder_op = encoder(embedded_user)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = embedded_user

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(10000):
        sess.run([optimizer, loss], feed_dict={user_inputs: inputt})
    print sess.run(embedded_user, feed_dict={user_inputs: tests})
    print sess.run(decoder_op, feed_dict={user_inputs: tests})
