import os
import re
import sys

import numpy as np


# Please modify to fit your environment
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import add, Input, Dense, Dropout, Add
from tensorflow.contrib.keras.api.keras import regularizers

import macro as mc

def latent(data_shape):
    model = Sequential()
    model.add(Dense(mc._OUT_DIM, activation='relu', input_shape=(data_shape,),\
    kernel_regularizer=regularizers.l2(mc._L2_REGULARIZE_RATE)))
    model.add(Dense(mc._OUT_DIM, activation='relu', input_shape=(data_shape,),\
    kernel_regularizer=regularizers.l2(mc._L2_REGULARIZE_RATE)))
    return model

def source_last_layer():
    model = Sequential()
    model.add(Dense(mc._SOURCE_DIM_NUM, name='source_nn_output', input_shape=(mc._OUT_DIM,)))
    return model

def target_last_layer():
    model = Sequential()
    model.add(Dense(mc._TARGET_DIM_NUM, name='target_nn_output', input_shape=(mc._OUT_DIM,)))
    return model
