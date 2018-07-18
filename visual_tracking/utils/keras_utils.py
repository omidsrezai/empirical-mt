import keras.backend as K
import tensorflow as tf

from keras.constraints import Constraint

tf_sess = tf.Session()
K.set_session(tf_sess)


class NonPos(Constraint):
    """Constrains the weights to be non-positive.
    """

    def __call__(self, w):
        w *= K.cast(K.less_equal(w, 0.), K.floatx())
        return w