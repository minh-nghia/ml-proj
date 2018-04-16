import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import roc_auc_score, average_precision_score

tf.set_random_seed(2018)


class AEOneClassSVM(object):
    """" AE-1SVM model. """
    def __init__(self, input_tensor, batch_size, name, autoencoder_layers, nu, alpha=1.0, sigma=1.0,
                 kernel_approx_features=1000,
                 autoencoder_activation='linear',
                 ae_op=tf.train.AdamOptimizer(1e-2),
                 full_op=tf.train.AdamOptimizer(1e-2), svm_op=tf.train.AdamOptimizer(1e-4), seed=2018):
        """ Constructor

        Args:
            input_tensor: The input tensor
            batch_size: Batch size
            name: Name of model. Will be used as variable scope.
            autoencoder_layers: Sizes of layers of the encoder, excluding the input layer.
            nu: Nu parameter of OC-SVM
            alpha: Alpha parameter of trade-off between encoding and SVM optimizing.
            sigma: Standard deviation of Random features.
            kernel_approx_features: Number of random features.
            autoencoder_activation: Activation function (linear, sigmoid, tanh, relu)
            ae_op: Autoencoder optimizer, default Adam(0.01)
            full_op: End-to-end optimizer, default Adam(0.01)
            svm_op: SVM optimizer, default Adam(0.0001)
            seed: Set TensorFlow seed.
        """

        self.input_tensor = input_tensor
        self.batch_size = batch_size
        self.name = name
        tf.set_random_seed(seed)
        self.seed = seed

        self.autoencoder_scope = '{}_autoencoder'.format(self.name)
        self.svm_scope = '{}_svm'.format(self.name)
        self.feature_analyser_scope = '{}_feature_analyser'.format(self.name)

        if autoencoder_activation == 'linear':
            self.autoencoder_activation = lambda x: tf.add(x, 0.0)
        elif autoencoder_activation == 'sigmoid':
            self.autoencoder_activation = lambda x: tf.sigmoid(x)
        elif autoencoder_activation == 'tanh':
            self.autoencoder_activation = lambda x: tf.tanh(x)
        elif autoencoder_activation == 'relu':
            self.autoencoder_activation = lambda x: tf.nn.relu(x)
        else:
            raise Exception()

        if autoencoder_layers is not None:
            self.bottleneck_size = autoencoder_layers[-1]
            self.encoded = self._create_encoder(self.input_tensor, autoencoder_layers)
            self.decoded = self._create_decoder(autoencoder_layers[::-1], self.input_tensor)
            self.reconstruction_loss = tf.reduce_mean(tf.pow(self.decoded - self.input_tensor, 2))
        else:
            self.encoded = input_tensor
            self.bottleneck_size = input_tensor.get_shape().as_list()[1]
            self.reconstruction_loss = 0

        self.alpha = alpha
        self.sigma = sigma
        self.nu = nu
        self.kernel_approx_features = kernel_approx_features
        (self.output, self.svm_loss, self.svm_weights,
         self.svm_rho, self.margin, self.rff_x) = self._create_ocsvm()

        self.gradient = tf.gradients(self.margin, self.input_tensor)
        self.gradient = tf.add(self.gradient, 0, name='gradient')  # Cannot assign tf.gradients to name
        self.input_x_gradient = tf.multiply(self.gradient, self.input_tensor, name='input_x_gradient')
        self.gradient_percent = tf.divide(self.input_x_gradient, tf.reduce_sum(self.input_x_gradient),
                                          name='gradient_percent')

        self.loss = self.alpha * self.reconstruction_loss + self.svm_loss
        full_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}|{}'.format(self.autoencoder_scope,
                                                                                       self.svm_scope))
        self.full_optimizer = full_op.minimize(self.loss, var_list=full_vars)

        if self.reconstruction_loss != 0:
            self.ae_optimizer = ae_op.minimize(
                self.reconstruction_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                     self.autoencoder_scope))
        self.svm_optimizer = svm_op.minimize(
            self.svm_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.svm_scope))

    def fit(self, sess, train_data, epochs_1, epochs_2, input_tensor=None, shuffle=False, verbose=False,
            validation_data=None, validation_label=None):
        """ Optimize full model with defined number of epochs.

        Args:
            sess: TF session
            train_data: Data.
            epochs_1: End-to-end training.
            epochs_2: SVM calibrating phase.
            input_tensor: Custom input tensor if needed.
            shuffle: Boolean indicating if data is shuffle each epoch.
            verbose: If verbose, print AUROC and AUPRC according to validation data and label.
            validation_data: Validation data
            validation_label: Validation label
        """

        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(train_data) / self.batch_size)

        data = train_data

        print('Combined train')
        for i in range(epochs_1):
            if shuffle:
                data = sk_shuffle(train_data, random_state=self.seed)
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                sess.run([self.full_optimizer, self.loss], feed_dict={input_tensor: batch_data})
            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: train_data}) / len(train_data)
            epoch_loss_ae = sess.run(self.reconstruction_loss, feed_dict={input_tensor: train_data}) / len(train_data)
            epoch_loss_svm = sess.run(self.svm_loss, feed_dict={input_tensor: train_data}) / len(train_data)
            if verbose:
                predictions = sess.run(self.output, feed_dict={input_tensor: validation_data})
                print('Epoch:', i + 1, 'Loss:', epoch_loss, '(', epoch_loss_ae, 'x', self.alpha, '+', epoch_loss_svm,
                      ')',
                      'AUROC:', roc_auc_score(validation_label, predictions),
                      'AUPRC:', average_precision_score(-validation_label, -predictions))
            else:
                print('.', end='')

        print('SVM train')
        for i in range(epochs_2):
            if shuffle:
                data = sk_shuffle(train_data, random_state=self.seed)
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                sess.run([self.svm_optimizer, self.svm_loss], feed_dict={input_tensor: batch_data})
            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: train_data}) / len(train_data)
            epoch_loss_ae = sess.run(self.reconstruction_loss, feed_dict={input_tensor: train_data}) / len(train_data)
            epoch_loss_svm = sess.run(self.svm_loss, feed_dict={input_tensor: train_data}) / len(train_data)
            if verbose:
                predictions = sess.run(self.output, feed_dict={input_tensor: validation_data})
                print('Epoch:', i + 1, 'Loss:', epoch_loss, '(', epoch_loss_ae, 'x', self.alpha, '+', epoch_loss_svm,
                      ')',
                      'AUROC:', roc_auc_score(validation_label, predictions),
                      'AUPRC:', average_precision_score(-validation_label, -predictions))
            else:
                print('.', end='')

    def fit_ae(self, sess, train_data, epochs, input_tensor=None, shuffle=False, verbose=False):
        """Train the autoencoder only."""
        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(train_data) / self.batch_size)

        data = train_data

        print('Autoencoder train')
        for i in range(epochs):
            if shuffle:
                data = sk_shuffle(train_data, random_state=self.seed)
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                sess.run([self.ae_optimizer, self.reconstruction_loss], feed_dict={input_tensor: batch_data})
            if verbose:
                epoch_loss = sess.run(self.reconstruction_loss, feed_dict={input_tensor: train_data}) / len(train_data)
                print('Epoch:', i + 1, 'Loss:', epoch_loss)
            else:
                print('.', end='')

    def fit_svm(self, sess, train_data, epochs, input_tensor=None, shuffle=False,
                verbose=False, validation_data=None, validation_label=None):
        """Train autoencoder only. See function fit for descriptions of verbose."""
        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(train_data) / self.batch_size)

        data = train_data

        print('SVM train')
        for i in range(epochs):
            if shuffle:
                data = sk_shuffle(train_data, random_state=self.seed)
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                sess.run([self.svm_optimizer, self.svm_loss], feed_dict={input_tensor: batch_data})
            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: train_data}) / len(train_data)
            epoch_loss_svm = sess.run(self.svm_loss, feed_dict={input_tensor: train_data}) / len(train_data)
            if verbose:
                predictions = sess.run(self.output, feed_dict={input_tensor: validation_data})
                print('Epoch:', i + 1, 'Loss:', epoch_loss_svm,
                      'AUROC:', roc_auc_score(validation_label, predictions),
                      'AUPRC:', average_precision_score(-validation_label, -predictions)
                      )

    def train_full(self, sess, data, input_tensor=None, dt=1e-10, limit=1000):
        """ Optimize full model with Early stopping.

        Args:
            sess: TF session.
            data: Data
            input_tensor: Custom input tensor.
            dt: Condition of termination. Stop training if change in loss < dt.
            limit: Maximum number of epochs.
        """
        print('Train full')
        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(data) / self.batch_size)
        i = 0
        loss_delta = 100.0
        last_loss = 0.0

        while loss_delta >= dt and i < limit:
            i += 1
            epoch_loss = 0
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                op_, loss_ = sess.run([self.full_optimizer, self.loss], feed_dict={input_tensor: batch_data})
                epoch_loss += loss_

            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: data}) / len(data)
            loss_delta = abs(last_loss - epoch_loss)
            last_loss = epoch_loss
            print(last_loss)
        print(i, 'epochs')

    def train_autoencoder(self, sess, data, input_tensor=None, dt=1e-10):
        print('Train AE')
        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(data) / self.batch_size)
        i = 0
        loss_delta = 100.0
        last_loss = 0.0

        while loss_delta >= dt:
            i += 1
            epoch_loss = 0
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                op_, loss_ = sess.run([self.ae_optimizer, self.loss], feed_dict={input_tensor: batch_data})
                epoch_loss += loss_

            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: data}) / len(data)
            loss_delta = abs(last_loss - epoch_loss)
            last_loss = epoch_loss
            print(last_loss)

    def calibrate_svm(self, sess, data, input_tensor=None, dt=1e-12, limit=1000):
        print('Calibrate SVM')
        input_tensor = input_tensor if input_tensor is not None else self.input_tensor
        batches = int(len(data) / self.batch_size)
        i = 0
        loss_delta = 100.0
        last_loss = 0.0

        while loss_delta >= dt and i < limit:
            i += 1
            epoch_loss = 0
            for b in range(batches):
                batch_data = data[b * self.batch_size:(b + 1) * self.batch_size]
                op_, loss_ = sess.run([self.svm_optimizer, self.loss], feed_dict={input_tensor: batch_data})
                epoch_loss += loss_

            epoch_loss = sess.run(self.loss, feed_dict={input_tensor: data}) / len(data)
            loss_delta = abs(last_loss - epoch_loss)
            last_loss = epoch_loss
            print(last_loss)
        print(i, 'epochs')

    def predict(self, sess, data):
        pred = sess.run(self.output, feed_dict={self.input_tensor: data}).T[0]

        return np.array([p if p != 0 else 1 for p in pred]).T

    def decision_function(self, sess, data):
        return sess.run(self.margin, feed_dict={self.input_tensor: data})

    def encode(self, sess, data):
        return sess.run(self.encoded, feed_dict={self.input_tensor: data})

    def encode_rff(self, sess, data):
        return sess.run(self.rff_x, feed_dict={self.input_tensor: data})

    def gradient(self, sess, data):
        return sess.run(self.gradient, feed_dict={self.input_tensor: data})

    def _create_encoder(self, x, hidden_nums):
        with tf.variable_scope(self.autoencoder_scope):
            layers = [None] * len(hidden_nums)

            # Xavier's Init
            init_bound = 4 * np.sqrt(6. / (x.get_shape().as_list()[1] + hidden_nums[0]))
            w0 = tf.Variable(tf.random_uniform([x.get_shape().as_list()[1], hidden_nums[0]], -init_bound, init_bound))
            b0 = tf.Variable(tf.random_uniform([hidden_nums[0]], -0.1, 0.1))
            layers[0] = self.autoencoder_activation(tf.add(tf.matmul(x, w0), b0))

            if len(layers) > 1:
                for i in range(1, len(layers)):
                    init_bound = 4 * np.sqrt(6. / (hidden_nums[i - 1] + hidden_nums[i]))
                    w = tf.Variable(tf.random_uniform([hidden_nums[i - 1], hidden_nums[i]], -init_bound, init_bound))
                    b = tf.Variable(tf.random_uniform([hidden_nums[i]], -0.1, 0.1))
                    layers[i] = tf.convert_to_tensor(
                        self.autoencoder_activation(tf.add(tf.matmul(layers[i - 1], w), b)))

        return layers[-1]

    def _create_decoder(self, hidden_nums, ae_input):
        with tf.variable_scope(self.autoencoder_scope):
            layers = [None] * len(hidden_nums)

            if len(layers) > 1:
                init_bound = 4 * np.sqrt(6. / (hidden_nums[0] + hidden_nums[1]))
                w0 = tf.Variable(tf.random_uniform([hidden_nums[0], hidden_nums[1]], -init_bound, init_bound))
                b0 = tf.Variable(tf.random_uniform([hidden_nums[1]], -0.1, 0.1))
                layers[0] = self.autoencoder_activation(tf.add(tf.matmul(self.encoded, w0), b0))

                for i in range(1, len(hidden_nums) - 1):
                    init_bound = 4 * np.sqrt(6. / (hidden_nums[i] + hidden_nums[i + 1]))
                    w = tf.Variable(tf.random_uniform([hidden_nums[i], hidden_nums[i + 1]], -init_bound, init_bound))
                    b = tf.Variable(tf.random_uniform([hidden_nums[i + 1]], -0.1, 0.1))
                    layers[i] = self.autoencoder_activation(tf.add(tf.matmul(layers[i - 1], w), b))

            output_dim = ae_input.get_shape().as_list()[1]
            w_out = tf.Variable(tf.truncated_normal([hidden_nums[-1], output_dim]))
            b_out = tf.Variable(tf.truncated_normal([output_dim]))
            if len(layers) > 1:
                layers[-1] = self.autoencoder_activation(tf.add(tf.matmul(layers[-2], w_out), b_out))
            else:
                layers[-1] = self.autoencoder_activation(tf.add(tf.matmul(self.encoded, w_out), b_out))

        return layers[-1]

    def _create_ocsvm(self):
        with tf.variable_scope(self.svm_scope):
            np.random.seed(self.seed)
            omega_matrix_shape = [self.bottleneck_size, self.kernel_approx_features]

            omega_matrix = tf.constant(
                np.random.normal(
                    scale=1.0 / self.sigma, size=omega_matrix_shape),
                dtype=tf.float32)
            omega_x = tf.matmul(self.encoded, omega_matrix)
            cos_omega_x = tf.cos(omega_x)
            sin_omega_x = tf.sin(omega_x)
            x = tf.sqrt(1.0 / self.kernel_approx_features) * tf.concat([cos_omega_x, sin_omega_x], 1)

            features = x.get_shape().as_list()[1]
            w = tf.Variable(tf.truncated_normal([features, 1]))
            rho = tf.Variable(0.0, tf.float32)

            n_margin = tf.subtract(rho, tf.matmul(x, w), name='margin'.format(self.name))

            reg_loss = 0.5 * tf.reduce_sum(tf.square(w))
            hinge_loss = tf.reduce_mean(n_margin + tf.abs(n_margin))

            total_loss = tf.subtract(reg_loss, rho) + tf.divide(hinge_loss, self.nu)
            output = tf.sign(-n_margin, name='output'.format(self.name))

            return output, total_loss, w, rho, -n_margin, x
