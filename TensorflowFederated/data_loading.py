from typing import Any, Optional, List
import tensorflow_federated as tff
from utils.data_utils import load_gen_data_as_train_test_dataset, preprocess


class GenDataBackend(tff.framework.DataBackend):

    def __init__(self,train_dataset,test_dataset,local_epochs):
        self.train_dataset= train_dataset
        self.test_dataset = test_dataset
        print("Loading data backend dataset of client has num of examples",self.train_dataset.cardinality())
        self.local_epochs = local_epochs

    # def preprocess(self,dataset : tf.data.Dataset):
    #     return dataset.shuffle(, seed =1,reshuffle_each_iteration=True).batch(BATCH).repeat(EPOCHS)

    async def materialize(self, data, type_spec):
        if data.uri[0] == "e":
            return preprocess(self.test_dataset,epochs=1)
        else:
            return preprocess(self.train_dataset,epochs=self.local_epochs)


class FederatedData(tff.program.FederatedDataSource,
                    tff.program.FederatedDataSourceIterator):
    """Interface for interacting with the federated training data."""

    def __init__(self, type_spec: tff.FederatedType):
        self._type_spec = type_spec
        self._capabilities = [tff.program.Capability.RANDOM_UNIFORM]

    @property
    def federated_type(self) -> tff.FederatedType:
        return self._type_spec

    @property
    def capabilities(self) -> List[tff.program.Capability]:
        return self._capabilities

    def iterator(self) -> tff.program.FederatedDataSourceIterator:
        return self

    def select(self, num_clients: Optional[int] = None) -> Any:
        data_uris = [f'uri://{i}' for i in range(num_clients)]
        return tff.framework.CreateDataDescriptor(
            arg_uris=data_uris, arg_type=self._type_spec)
