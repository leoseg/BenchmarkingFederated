from keras.metrics import AUC
from keras.utils import to_categorical
import tensorflow as tf

class SparseAUC(AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        return super().update_state(y_true,y_pred,sample_weight)