from typing import Any, Optional, List
import tensorflow as tf
import tensorflow_federated as tff
from utils.data_utils import load_gen_data

GEN_DATAPATH = "/app/data/Dataset.csv"
class GenDataBackend(tff.framework.DataBackend):
    def __int__(self):
        super().__init__()
        self.shuffle = 100
        self.batch = 20

    async def materialize(self, data, type_spec):
        df = tf.convert_to_tensor(load_gen_data(GEN_DATAPATH))
        client_dataset = tf.data.Dataset.from_tensor_slices((df[:, :-1], df[:, -1])).shuffle(self.shuffle).batch(
            self.batch)
        return client_dataset




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