
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Layer
# from tf.keras import backend as K


class MultiDense(Layer):
    """Multiple Dense layer without biases
    Returns
    sum_i x[i] @ w[i]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # multiply one parameter to an identity matrix and then multiply it with a variable
    def build(self, input_shape):
        self.d = input_shape[0][-1]
        num_inputs = len(input_shape)
        self.kernels = []
        for i in range(num_inputs):
            self.kernels.append(self.add_weight(name=f'kernel_{i}',
                                shape=(1, 1),
                                initializer='uniform',
                                trainable=True))
        super().build(input_shape)

    def call(self, x):
        self.kernels_matrix = []
        E = tf.eye(self.d, dtype='float64')
        for j in range(len(self.kernels)):
            matrix = tf.multiply(self.kernels[j],E)
            self.kernels_matrix.append(matrix)

        return tf.add_n([tf.matmul(xi, wi) for xi, wi in zip(x, self.kernels_matrix)])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
 

class SoftmaxDense(Layer):
    """
    Softmax on the input to ensure the sum of last layer is 1
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.d = input_shape[0][-1]
        num_inputs = len(input_shape)
        self.kernels = []
        for i in range(num_inputs):
            self.kernels.append(self.add_weight(name=f'kernel_{i}',
                                shape=(1, 1),
                                initializer = 'uniform',
                                trainable=True))
        super().build(input_shape)

    def call(self, x):
        w_stack = tf.stack(self.kernels, axis=0)
        w_softmax = tf.nn.softmax(w_stack, axis=0)
        w_unstacked = tf.unstack(w_softmax, axis=0)

        self.kernels_matrix = []
        E = tf.eye(self.d, dtype='float64')
        for j in range(len(self.kernels)):
            matrix = tf.multiply(w_unstacked[j],E)
            self.kernels_matrix.append(matrix)

        return tf.add_n([tf.matmul(xi, wi) for xi, wi in zip(x, self.kernels_matrix)])
