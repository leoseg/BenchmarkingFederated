import tensorflow as tf
import tensorflow_federated as tff
from keras.utils import set_random_seed

from utils.data_utils import df_train_test_dataset, preprocess


class DataBackend(tff.framework.DataBackend):
    """
    Custom databackend that materialize and preprocesses dataset for federated learning
    """

    def __init__(self,train_dataset,test_dataset,local_epochs):
        self.train_dataset= train_dataset
        self.test_dataset = test_dataset
        print("Loading data backend dataset of client has num of examples",self.train_dataset.cardinality())
        self.local_epochs = local_epochs

    # def preprocess(self,dataset : tf.data.Dataset):
    #     return dataset.shuffle(, seed =1,reshuffle_each_iteration=True).batch(BATCH).repeat(EPOCHS)

    async def materialize(self, data, type_spec):
        tf.print(f"Materializing data for client {data.uri}"
                 f"Train dataset has size {self.train_dataset.cardinality()}",
                 f"Test dataset has size {self.test_dataset.cardinality()}")
        #for i in range(0,5):
        tf.print(f"train dataset entry {0} from client {data.uri[-1]} is {list(self.train_dataset.as_numpy_iterator())[0]}")
        tf.print(f"test dataset entry {0} from client {data.uri[-1]} is {list(self.test_dataset.as_numpy_iterator())[0]}")
        if data.uri[0] == "e":
            return self.test_dataset.batch(32)
        else:
            tf.print(f"epochs are {self.local_epochs}")
            set_random_seed(1)
            preprocessed_ds =  preprocess(self.train_dataset,epochs=self.local_epochs)
            tf.print(f"preprocessed dataset entry {0} from client {data.uri[-1]} is {list(preprocessed_ds.as_numpy_iterator())[0]}")
            return preprocessed_ds


# class FederatedData(tff.program.FederatedDataSource,
#                     tff.program.FederatedDataSourceIterator):
#     """Interface for interacting with the federated training data."""
#
#     def __init__(self, type_spec: tff.FederatedType):
#         self._type_spec = type_spec
#         self._capabilities = [tff.program.Capability.RANDOM_UNIFORM]
#
#     @property
#     def federated_type(self) -> tff.FederatedType:
#         return self._type_spec
#
#     @property
#     def capabilities(self) -> List[tff.program.Capability]:
#         return self._capabilities
#
#     def iterator(self) -> tff.program.FederatedDataSourceIterator:
#         return self
#
#     def select(self, num_clients: Optional[int] = None) -> Any:
#         data_uris = [f'uri://{i}' for i in range(num_clients)]
#         return tff.framework.CreateDataDescriptor(
#             arg_uris=data_uris, arg_type=self._type_spec)
