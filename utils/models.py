from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1_l2

param_num_nodes = 1024
param_dropout_rate = 0.3
param_l1_v = 0.0
param_l2_v = 0.005


def get_seq_nn_model(input_dim:int,num_nodes: int = param_num_nodes, dropout_rate: float = param_dropout_rate, l1_v: float = param_l1_v,
                     l2_v: float = param_l2_v):
    """
    Creates utils with given parameters
    :param input_dim: input dimension of data
    :param num_nodes: number of nodes in dense layers
    :param dropout_rate: dropout rate after denser layer
    :param l1_v: l1 kernel regularizer
    :param l2_v: l2 kernel regularizer
    :return: uncompiled utils
    """
    model = Sequential()
    # input layer
    model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.0, l2=0.0), input_dim=input_dim))
    model.add(Dropout(0.4))

    # first layer
    model.add(Dense(num_nodes, activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # second layer
    model.add(Dense(int(num_nodes / 2), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # third layer
    model.add(Dense(int(num_nodes / 2), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # fourth layer
    model.add(Dense(int(num_nodes / 4), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # fifth layer
    model.add(Dense(int(num_nodes / 4), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # sixth layer
    model.add(Dense(int(num_nodes / 8), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # seventh layer
    model.add(Dense(int(num_nodes / 8), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    # eighth layer
    model.add(Dense(int(num_nodes / 16), activation='relu', kernel_regularizer=l1_l2(l1=l1_v, l2=l2_v), input_dim=input_dim))
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(units=1, activation="tanh"))
    return model
