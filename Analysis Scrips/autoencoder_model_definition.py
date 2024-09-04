# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:09:40 2023

@author: Utente
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, regularizers, Model
from keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam

seed=42
tf.random.set_seed(seed)


def build_model(input_size=170, h_dim=[75], z_dim=60, activation_function='selu', activation_output='linear',
                output_initializer=tf.keras.initializers.glorot_uniform(seed=seed), kernel_initial='lecun_normal',
                kernel_regul=0.0001, learnRate=0.001, loss_function=keras.losses.MeanSquaredError(), metric=['mse']):

    #Encoder part
    input_layer = Input(shape=(input_size,))
    
    x=input_layer
    
    # Hidden layers
    for n_neurons_layer in h_dim:
        
        x=Dense(n_neurons_layer, activation=activation_function,kernel_initializer=kernel_initial, 
                kernel_regularizer=regularizers.l2(kernel_regul))(x)
        
    encoded= Dense(z_dim ,activation=activation_function,kernel_initializer=kernel_initial, 
                   kernel_regularizer=regularizers.l2(kernel_regul))(x)
    
    decoder_inputs=Input(shape=(z_dim,))
    
    x=decoder_inputs
    for n_neurons_layer in  reversed(h_dim):
        x=Dense(n_neurons_layer, activation=activation_function,kernel_initializer=kernel_initial, 
                kernel_regularizer=regularizers.l2(kernel_regul))(x)
    
    output_decoder=Dense(input_size, activation=activation_output, 
                         kernel_initializer=output_initializer)(x)
      
    # Encoder Block
    encoder=Model(inputs=input_layer, outputs=encoded, name='encoder')
    
    #Decoder Block
    decoder=Model(inputs=decoder_inputs, outputs=output_decoder, name='decoder')
  
    model = Model(inputs=input_layer, outputs= decoder(encoder(input_layer)), 
                  name="variational_AE")
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate= learnRate), loss=loss_function, metrics = metric)
    
    return model  #encoder, decoder,