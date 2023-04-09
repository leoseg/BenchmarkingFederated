import numpy as np
from tensorflow import keras
from evaluation_utils import evaluate_model, load_test_data_for_evaluation
# Load the saved weights
X_test,y_test = load_test_data_for_evaluation()
path_to_tff_weights = "../TensorflowFederated/tff_weights.h5"
path_to_flwr_weights = "../Flower/flwr_weights.h5"
evaluate_model(path_to_flwr_weights, X_test,y_test)
evaluate_model(path_to_tff_weights, X_test,y_test)
