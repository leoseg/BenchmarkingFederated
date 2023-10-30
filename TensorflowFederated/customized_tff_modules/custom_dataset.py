import tensorflow as tf


class labeled_dataset(tf.data.Dataset):
    def assign_label(self, label):
        self.label = label
