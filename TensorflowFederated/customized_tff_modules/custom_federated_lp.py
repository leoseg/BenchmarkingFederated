import keras.losses
import tensorflow_federated as tff
import tensorflow as tf
import keras.backend as K
from keras.metrics import AUC, Precision, Recall, Accuracy


def create_iterativ_procss(keras_model_fn, input_dim, element_spec):
    def model_fn():
        keras_model = keras_model_fn(input_dim)
        keras_model.trainable = True
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=element_spec,
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[Accuracy(), AUC(curve="PR"), Precision(), Recall()],
        )

    @tff.tf_computation
    def server_init():
        model = model_fn()
        return model.trainable_variables

    model_weights_type = server_init.type_signature.result
    model_for_spec = model_fn()
    tf_dataset_type = tff.SequenceType(model_for_spec.input_spec)

    @tf.function
    def client_update(model, dataset, server_weights, client_optimizer):
        """Performs training (using the server model weights) on the client's dataset."""
        # Initialize the client model with the current server weights.
        client_weights = model.trainable_variables
        # Assign the server weights to the client model.
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

        # Use the client_optimizer to update the local model.
        for batch in dataset:
            with tf.GradientTape() as tape:
                # Compute a forward pass on the batch of data
                outputs = model.forward_pass(batch)

            tf.print(outputs.loss)
            # Compute the corresponding gradient
            grads = tape.gradient(outputs.loss, client_weights)
            grads_and_vars = zip(grads, client_weights)
            tf.print(
                K.mean(K.equal(batch[1], K.cast(K.round(outputs.predictions), "int64")))
            )
            # Apply the gradient using a client optimizer.
            client_optimizer.apply_gradients(grads_and_vars)

        return client_weights

    @tf.function
    def server_update(model, mean_client_weights):
        """Updates the server model weights as the average of the client model weights."""
        model_weights = model.trainable_variables
        # Assign the mean client weights to the server model.
        tf.nest.map_structure(
            lambda x, y: x.assign(y), model_weights, mean_client_weights
        )
        return model_weights

    @tff.tf_computation(tf_dataset_type, model_weights_type)
    def client_update_fn(tf_dataset, server_weights):
        model = model_fn()
        client_optimizer = tf.keras.optimizers.Adam()
        return client_update(model, tf_dataset, server_weights, client_optimizer)

    @tff.tf_computation(model_weights_type)
    def server_update_fn(mean_client_weights):
        model = model_fn()
        return server_update(model, mean_client_weights)

    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value(server_init(), tff.SERVER)

    federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
    federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

    @tff.federated_computation(federated_server_type, federated_dataset_type)
    def next_fn(server_weights, federated_dataset):

        # Broadcast the server weights to the clients.
        server_weights_at_client = tff.federated_broadcast(server_weights)

        # Each client computes their updated weights.
        client_weights = tff.federated_map(
            client_update_fn, (federated_dataset, server_weights_at_client)
        )

        # The server averages these updates.
        mean_client_weights = tff.federated_mean(client_weights)

        # The server updates its model.
        server_weights = tff.federated_map(server_update_fn, mean_client_weights)

        return server_weights

    federated_algorithm = tff.templates.IterativeProcess(
        initialize_fn=initialize_fn, next_fn=next_fn
    )
    return federated_algorithm
