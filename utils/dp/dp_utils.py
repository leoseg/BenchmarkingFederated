import math
import sys
from typing import List

import keras.callbacks
import numpy as np
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import tensorflow as tf


BETA = math.exp(-0.5)  # from paper
BITS = 20
DIV_EPSILON = 1e-22
GAMMA = 1e-10


def create_compression_sum_query(
    l2_norm_bound, ddp_query, client_template=None, conditional=True, beta=BETA
):
    """
    This function creates a compression sum query which quantizes the input data and then uses the given
    distributed differential privacy query to add noise to the quantized data.
    """
    scale = 1.0 / (GAMMA + DIV_EPSILON)
    quantization_params = compression_query.QuantizationParams(
        stochastic=True,
        conditional=conditional,
        l2_norm_bound=l2_norm_bound,
        beta=beta,
        quantize_scale=scale,
    )
    quantized_ddp_query = compression_query.CompressionSumQuery(
        quantization_params=quantization_params,
        inner_query=ddp_query,
        record_template=client_template,
    )
    return quantized_ddp_query


def calculate_delta(num_examples: int) -> float:
    """
    This function calculates an adequate delta for differential privacy in machine learning.

    Parameters:
    num_examples (int): The number of examples.

    Returns:
    float: The calculated delta.
    """
    # Calculate inverse of the number of examples
    inverse = 1 / num_examples

    # Calculate the number of decimal places in the inverse
    decimal_places = abs(math.floor(math.log10(inverse)))

    # Calculate delta as 1 divided by 10 to the power of (decimal places + 1)
    delta = 1 / (10 ** (decimal_places + 1))

    return delta


def calculate_epsilon_for_usecases(noise_multiplier):
    """
    This function calculates the epsilon for the usecases 1, 2, 3 and 4.
    :param noise_multiplier: noise multiplier to use for calculation
    :return:
    """
    usecase_data = {
        1: {
            "batch_size": 512,
            "num_examples_list": [3207],  # , 9261],# 192],
            "epochs": 70,
        },
        2: {
            "batch_size": 512,
            "num_examples_list": [3207],  # , 9261],# 192],
            "epochs": 8,
        },
        3: {
            "batch_size": 512,
            "num_examples_list": [1848],  # , 5544],#, 110],
            "epochs": 100,
        },
        4: {
            "batch_size": 512,
            "num_examples_list": [1848],  # , 5544],#, 110],
            "epochs": 30,
        },
    }
    for use_case in [1, 2, 3, 4]:
        data = usecase_data[use_case]
        print("-------------------Usecase {}-------------------".format(use_case))
        for num_examples in data["num_examples_list"]:
            delta = calculate_delta(num_examples)
            if data["batch_size"] > num_examples:
                data["batch_size"] = num_examples

            print(
                compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                    n=num_examples,
                    batch_size=data["batch_size"],
                    noise_multiplier=noise_multiplier,
                    epochs=data["epochs"],
                    delta=delta,
                )
            )


class GlobalNorm(keras.callbacks.Callback):

    median_global_norm = 0

    batch_global_norms = []

    def on_train_begin(self, logs=None):
        self.batch_global_norms = []
        self.median_global_norm = 0

    def on_train_batch_end(self, batch, logs=None):
        weights = self.model.get_weights()
        flatten_weights = tf.nest.flatten(weights)
        global_norm = tf.linalg.global_norm(flatten_weights)
        self.batch_global_norms.append(global_norm)

    def on_train_end(self, logs=None):
        self.median_global_norm = np.median(np.asarray(self.batch_global_norms))


if __name__ == "__main__":
    calculate_epsilon_for_usecases(float(sys.argv[1]))
