import tensorflow as tf
import tensorflow_federated as tff
from keras.utils import set_random_seed

from utils.data_utils import df_train_test_dataset, preprocess


class DataBackend(tff.framework.DataBackend):
    """
    Custom databackend that materialize and preprocesses dataset for federated learning
    """

    def __init__(self, train_dataset, test_dataset, local_epochs):
        """
        Init function for DataBackend
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print(
            "Loading data backend dataset of client has num of examples",
            self.train_dataset.cardinality(),
        )
        self.local_epochs = local_epochs

    async def materialize(self, data, type_spec):
        """
        Materializes the data on client
        """
        tf.print(
            f"Materializing data for client {data.uri}"
            f"Train dataset has size {self.train_dataset.cardinality()}",
            f"Test dataset has size {self.test_dataset.cardinality()}",
        )
        # tf.print(f"train dataset entry {0} from client {data.uri[-1]} is {list(self.train_dataset.as_numpy_iterator())[0]}")
        # tf.print(f"test dataset entry {0} from client {data.uri[-1]} is {list(self.test_dataset.as_numpy_iterator())[0]}")
        if data.uri[0] == "e":
            return self.test_dataset.batch(32)
        else:
            tf.print(f"epochs are {self.local_epochs}")
            preprocessed_ds = preprocess(
                self.train_dataset, epochs=self.local_epochs, seed=int(data.uri[0])
            )
            tf.print(
                f"preprocessed dataset entry {0} from client {data.uri[-1]} is {list(preprocessed_ds.as_numpy_iterator())[0]}"
            )
            return preprocessed_ds
