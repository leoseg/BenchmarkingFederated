import collections

from utils.data_utils import load_gen_data_as_train_test_split
import tensorflow as tf
import tensorflow_federated as tff
from utils.models import get_seq_nn_model
from keras.metrics import AUC, Precision, Recall

# emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
GEN_DATAPATH = "../DataGenExpression/Dataset1.csv"

NUM_EPOCHS = 100
SHUFFLE_BUFFER = 100
configs = dict(
    batch_size=512,
    epochs=100,
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(curve="PR"), Precision(), Recall()],
    earlystopping_patience=5,
    num_nodes=1024,
    dropout_rate=0.3,
    l1_v=0.0,
    l2_v=0.005,
    n_splits=5,
)


# example_dataset = emnist_train.create_tf_dataset_for_client(
#   emnist_train.client_ids[0])

# df = load_gen_data(GEN_DATAPATH)
# X, y = create_X_y(df)
X_train, X_test, y_train, y_test = load_gen_data_as_train_test_split(
    data_path=GEN_DATAPATH
)
client_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000, reshuffle_each_iteration=True)
    .batch(512)
    .repeat(NUM_EPOCHS)
)


class GenDataBackend(tff.framework.DataBackend):
    def __int__(self):
        super().__init__()

    def preprocess(self, dataset: tf.data.Dataset):
        return dataset.batch(512)  # .repeat(NUM_EPOCHS)

    async def materialize(self, data, type_spec):
        client_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10000, reshuffle_each_iteration=True)
            .batch(512)
            .repeat(NUM_EPOCHS)
        )
        return client_dataset


def model_fn():
    keras_model = get_seq_nn_model(
        12708,
        configs.get("num_nodes"),
        configs.get("dropout_rate"),
        configs.get("l1_v"),
        configs.get("l2_v"),
    )
    keras_model.trainable = True
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=client_dataset.element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            AUC(curve="PR"),
            Precision(),
            Recall(),
        ],
    )


def evaluate(server_state):
    keras_model = get_seq_nn_model(
        12708,
        configs.get("num_nodes"),
        configs.get("dropout_rate"),
        configs.get("l1_v"),
        configs.get("l2_v"),
    )
    keras_model.compile(
        optimizer=configs.get("optimizer"),
        loss=configs.get("loss"),
        metrics=configs.get("metrics"),
    )
    server_state.assign_weights_to(keras_model)
    # keras_model.set_weights(server_state)
    keras_model.evaluate(X_test, y_test)


# X, y = create_X_y(df)
# test_dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(10000,reshuffle_each_iteration=True).batch(512).repeat(40)
# model = get_seq_nn_model(12708, configs.get("num_nodes"),configs.get("dropout_rate"), configs.get("l1_v"), configs.get("l2_v"))
# model.compile(optimizer=configs.get("optimizer"),
#                   loss=configs.get("loss"),
#                   metrics=configs.get("metrics"))
# print("WITH X,Y")
# model.fit(X,y, epochs=40 ,batch_size=512)
print("WITH TF DATASET")
model = get_seq_nn_model(
    12708,
    configs.get("num_nodes"),
    configs.get("dropout_rate"),
    configs.get("l1_v"),
    configs.get("l2_v"),
)
model.compile(
    optimizer=configs.get("optimizer"),
    loss=configs.get("loss"),
    metrics=configs.get("metrics"),
)
model.fit(client_dataset)


def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    # A `DataBackend` object is wrapped by a `DataExecutor`, which queries the
    # backend when a TFF worker encounters an operation requires fetching local
    # data.
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device), data_backend=GenDataBackend()
    )


# In a distributed setting, this needs to run in the TFF worker as a service
# connecting to some port. The top-level controller feeding TFF computations
# would then connect to this port.
factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)
ctx = tff.framework.ExecutionContext(executor_fn=factory)
tff.framework.set_default_context(ctx)

# iterative_process = create_iterativ_procss(get_seq_nn_model,12708,client_dataset.element_spec)

iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)


# evaluate(model.get_weights())
state = iterative_process.initialize()


NUM_CLIENTS = 1
element_type = tff.types.StructWithPythonType(
    client_dataset.element_spec, container_type=collections.OrderedDict
)
dataset_type = tff.types.SequenceType(element_type)

round_data_uris = [f"uri://{i}" for i in range(NUM_CLIENTS)]
round_train_data = tff.framework.CreateDataDescriptor(
    arg_uris=round_data_uris, arg_type=dataset_type
)

for i in range(0, 2):
    # state = iterative_process.next(state,[client_dataset])
    # custom
    # evaluate(state)
    # tff
    result = iterative_process.next(state, round_train_data)
    state = result.state
    model_weights = iterative_process.get_model_weights(state)
    evaluate(model_weights)
    metrics = result.metrics
    print(f"round {i}, train_metrics={metrics}")
