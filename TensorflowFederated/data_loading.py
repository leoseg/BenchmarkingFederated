from typing import Any, Optional, List
import tensorflow as tf
import tensorflow_federated as tff
from utils.data_utils import load_gen_data_as_train_test_split
from tff_config import EPOCHS,BATCH, N_ROWS
GEN_DATAPATH = "../DataGenExpression/Dataset1.csv"
SHUFFLE = 10000

class GenDataBackend(tff.framework.DataBackend):


    def __init__(self):
        X_train, X_test, y_train, y_test = load_gen_data_as_train_test_split(data_path=GEN_DATAPATH)
        self.client_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))




    def preprocess(self,dataset : tf.data.Dataset):
        return dataset.shuffle(SHUFFLE, seed =1,reshuffle_each_iteration=True).batch(BATCH).repeat(EPOCHS)

    async def materialize(self, data, type_spec):
        return self.preprocess(self.client_dataset)




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