import numpy as np
import tensorflow as tf

#TF Imports
from tensorflow.contrib.keras import layers, models, optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import initializers


class Actor:
    
    def __init__(self, state_size, action_size, action_low, action_high):
        """
            state_size: Dimension of each state
            action_size: Dimention of each action
            action_low, action_high: Min and Max values of action dimensions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self): 
        states = layers.Input(shape=(self.state_size,), name='inputStates')

        # Hidden Layers
        model = layers.Dense(units=128, activation='linear')(states)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(0.01)(model)
        model = layers.Dropout(0.3)(model)
        
        model = layers.Dense(units=256, activation='linear')(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(0.01)(model)
        model = layers.Dropout(0.3)(model)

        model = layers.Dense(units=512, activation='linear')(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(0.01)(model)
        model = layers.Dropout(0.3)(model)

        model = layers.Dense(units=128, activation='linear')(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(0.01)(model)
        model = layers.Dropout(0.3)(model)

        output = layers.Dense(
            units=self.action_size, 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(0.01),
            name='outputActions')(model)

        #Keras
        self.model = models.Model(inputs=states, outputs=output)

        #Definint Optimizer
        actionGradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-actionGradients * output)
        optimizer = optimizers.Adam()
        update_operation = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, actionGradients, K.learning_phase()],
            outputs=[], 
            updates=update_operation)