# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:29:40 2023

@author: Utente
"""
#Datasets and system
import numpy as np                # Management of arrays
import os                         # System utils
import pandas as pd
from pathlib import Path          # path and file utils
from scipy.io import loadmat      # Required to load .mat files
import h5py #For creating new dataset files
import sklearn 
import copy

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, History
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# INSERT PATH XXXX

import sys
sys.path.insert(0,"XXXX")
from basic_model import build_model
from hyperparameter_tunning import hyperparameter_tunning

import sys
sys.path.insert(0,"XXXX")
import confounder_correction_classes
from confounder_correction_classes import StandardScalerDict, BiocovariatesRegression
                

# Load data 

# HCP 
data_path="XXXX"

HCP_covars=pd.read_csv(os.path.join(data_path, "HCP_covar.csv"))
HCP_data=pd.read_csv(os.path.join(data_path,"HCP_data.csv"))
n_subj=HCP_data.shape[0]

HCP_data=HCP_data.to_numpy()
HCP_data=HCP_data[:,0:170] # Last features will not be used as they are cortical thickness but from another  ROI map

#HCP_data=K.eval(HCP_data)
#HCP_data = tf.convert_to_tensor(HCP_data, dtype=tf.float32)

# Brain features index-order definition
dk40_feat= np.arange(0,68) # 68 cortical thickness
cobravgm_feat=np.arange(68,118) # 52 gray matter volumes
cobravwm_feat=np.arange(118,170) # 52 white mater volumes
vol_feat=np.concatenate((cobravgm_feat,cobravwm_feat))

# Brain feature-types to consider separatly and associated bio-covariates 
feat_of_interest={'cortical': {'id': dk40_feat, 'categorical': ['Gender'], 
                               'continuous':['Age']},  
                  'volumes': {'id': vol_feat, 'categorical': ['Gender'],
                              'continuous':['Age','TIV']}}
  
# Preprocessing pipelne: Regressing-out confounding biological covariates
scaler_data_fun=StandardScalerDict(std_data=True, std_cov=False, output_dict=True)
biocov_regre=BiocovariatesRegression(cv_method=None,feat_detail=feat_of_interest)

# Defining the pipeline to pass inside hyperparameter tunning
steps_biocov=[('Standardize only data', scaler_data_fun),('Confounder Regression', biocov_regre),('Standardize', StandardScaler())]
pipeline_biocov_reg= Pipeline(steps_biocov)


# Random search iterations and see
random_search={'max_iter': 2000, 'seed':123}

# Path to Save results
path_save ="XXXX\\rand_search_trial3_s123.csv"
 
# Define hyperparameter space
# model__ : for hyperparameters that are defined when creating build_model
# optimizer_: for hyperparameters that should be passed inside compile optimizer
#             and have not been defined as model attributes
# nothing : for hyperparameters that will be defined inside KerasWrapper for training process

# Define a learning schedule to pass to learning rate
def lr_schedule_fun(initial_lr, batch, n_subjects):
    decay_step= n_subj/batch
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr,decay_steps=decay_step,decay_rate=0.9977,staircase=False)
    return lr_schedule

hyperparameter_space = {'model__h_dim':[[120,90,70],[120,100,80], [100,70], [120,80], [90], [100], [80]], 
                        'model__z_dim': [20,25,30,35, 40,45,50, 55,60], 
                        'model__learnRate':[0.00001, 0.000001,0.0001,0.005,0.00005,0.001,0.0005], #0.00001, 0.000001,0.0001,lr_schedule_fun(0.05, 32, n_subj)
                        'batch_size' : [32], # 32, 64, 128, 254
                        'epochs': [500,2000,1000],
                        #'loss':['mean_absolute_error','mean_squared_error']} #MSE
                        'model__kernel_regul':[0.001,0.0001,0.00001,0.005,0.0005,0.00005]}  #loguniform(1e-6, 1e-1)


# Perform search
search = hyperparameter_tunning(HCP_data, HCP_covars,pipeline_biocov_reg,random_search, build_model,
                              hyperparameter_space, path_save, verbose=1)

#search.loc[search[['RE_best_score']].idxmax(axis='rows')]


# Check best result

# Load csv files wit hyper results
hyper_path="C:\\Users\\Utente\\Documents\\Ines_W\\AE_normative"
hyper_result=pd.read_csv(os.path.join(hyper_path, "rand_search_trial2_s42.csv"))

hyper_result.loc[hyper_result[['RE_best_score']].idxmax(axis='rows')]



""""
#autoencoder.summary(expand_nested=True)

#train model 

encoder, decodor, autoencoder= build_model(input_size=170, h_dim=[100, 70], z_dim=50, activation_function='selu',activation_output='linear',
                output_initializer= tf.keras.initializers.glorot_uniform(), kernel_initial='lecun_normal',
                kernel_regul=0.0001, learnRate=0.0001, loss_function=keras.losses.MeanSquaredError(), metric= ['mse'])


""""