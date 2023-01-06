import os
import flwr as fl
from utils.models import get_seq_nn_model
from utils.data_utils import load_gen_data_as_train_test_split

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
datapath="app/data/Dataset.csv"
# Load model and data
(x_train, y_train), (x_test, y_test) = load_gen_data_as_train_test_split(datapath)
model = get_seq_nn_model(input_dim=x_train.shape[1])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
