import numpy as np
import tensorflow as tf

# Tensorflow layer imports
from tensorflow.contrib.keras import layers, models, optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import initializers


class Critic:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        #Define input layers
        inputStates = layers.Input(shape=(self.state_size,), name='inputStates')
        inputActions = layers.Input(shape=(self.action_size,), name='inputActions')

        # Hidden layers for states
        modelS = layers.Dense(units=128, activation='linear')(inputStates)
        modelS = layers.BatchNormalization()(modelS)
        modelS = layers.LeakyReLU(0.01)(modelS)
        modelS = layers.Dropout(0.3)(modelS)

        modelS = layers.Dense(units=256, activation='linear')(modelS)
        modelS = layers.BatchNormalization()(modelS)
        modelS = layers.LeakyReLU(0.01)(modelS)
        modelS = layers.Dropout(0.3)(modelS)
        
        modelA = layers.Dense(units=256, activation='linear')(inputActions)
        modelA = layers.LeakyReLU(0.01)(modelA)
        modelA = layers.BatchNormalization()(modelA)
        modelA = layers.Dropout(0.5)(modelA)
        
        #Merging the models
        model = layers.add([modelS, modelA])
        model = layers.Dense(units=256, activation='linear')(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(0.01)(model)

        #Q Layer
        Qvalues = layers.Dense(
            units=1, 
            activation=None, 
            name='outputQvalues')(model)

        #Keras model
        self.model = models.Model(inputs=[inputStates, inputActions], outputs=Qvalues)
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        actionGradients = K.gradients(Qvalues, inputActions)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=actionGradients)