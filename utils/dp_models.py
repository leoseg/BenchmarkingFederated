import keras
from keras.layers import Dense, Dropout,Input
from keras.models import Sequential
from keras.regularizers import l1_l2, l2
from keras.utils import set_random_seed
from utils.config import SEED
from config import configs
import tensorflow as tf


def get_model(**kwargs)->keras.Model:
    """
    Uses model depending on the configs
    :param **kwargs for model
    :return: model
    """
    set_random_seed(SEED)
    if configs.get("usecase") == 1:
        return get_seq_nn_model(input_dim=kwargs["input_dim"],num_nodes=kwargs["num_nodes"],dropout_rate=kwargs["dropout_rate"],l1_v=kwargs["l1_v"],l2_v=kwargs["l2_v"])
    elif configs.get("usecase") == 2:
        return get_log_reg_keras(l2_v=kwargs["l2_v"])
    elif configs.get("usecase") == 3:
        return get_log_reg_keras(l2_v=kwargs["l2_v"])
    elif configs.get("usecase") == 4:
        return get_seq_nn_model(input_dim=kwargs["input_dim"],num_nodes=kwargs["num_nodes"],dropout_rate=kwargs["dropout_rate"],l1_v=kwargs["l1_v"],l2_v=kwargs["l2_v"])






param_num_nodes = 1024
param_dropout_rate = 0.3
param_l1_v = 0.0
param_l2_v = 0.005
def get_seq_nn_model(input_dim:int,num_nodes: int = param_num_nodes, dropout_rate: float = param_dropout_rate, l1_v: float = param_l1_v,
                     l2_v: float = param_l2_v):
    """
    Creates model with given parameters
    :param input_dim: input dimension of data
    :param num_nodes: number of nodes in dense layers
    :param dropout_rate: dropout rate after denser layer
    :param l1_v: l1 kernel regularizer
    :param l2_v: l2 kernel regularizer
    :return: uncompiled utils
    """
    model = Sequential()
    # input layer
    model.add(Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) , input_dim=input_dim))
    model.add(Dropout(0.4,seed=1))

    # first layer
    model.add(Dense(num_nodes, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) , input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))
    # second layer
    model.add(Dense(int(num_nodes / 2), activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) ,  input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))
    # third layer
    model.add(Dense(int(num_nodes / 2), activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1), input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))
    # fourth layer
    model.add(Dense(int(num_nodes / 4), activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) ,  input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1 ))
    # fifth layer
    model.add(Dense(int(num_nodes / 4), activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) , input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1 ))
    # sixth layer
    model.add(Dense(int(num_nodes / 8), activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) ,input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))
    # seventh layer
    model.add(Dense(int(num_nodes / 8), activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) ,  input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))
    # eighth layer
    model.add(Dense(int(num_nodes / 16), activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1) , input_dim=input_dim))
    model.add(Dropout(dropout_rate,seed=1))

    # output layer
    model.add(Dense(units=configs["number_of_classes"], activation=configs["activation"]))
    return model


def get_log_reg_keras(l2_v):
    """
    Returns log regression model implemented in keras
    :param l2_v: regularization in l2
    :return: model
    """
    model = Sequential()
    #model.add(Input(shape=(configs["batch_size"],input_dim)))
    model.add(Dense(input_dim=configs["input_dim"],units=configs["number_of_classes"], kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1), activation=configs["activation"]))
    return model


