# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:11:13 2023

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


#!pip install scikeras
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

## INSERT PATH IN XXXX
import sys
sys.path.insert(0,"XXXX")
import confounder_correction_classes
from confounder_correction_classes import StandardScalerDict, BiocovariatesRegression

# Model
import sys
sys.path.insert(0,"XXXX")
from basic_model import build_model

# Load data 

# HCP 
data_path="XXXX"

HCP_covars=pd.read_csv(os.path.join(data_path, "HCP_covar.csv"))
HCP_data=pd.read_csv(os.path.join(data_path,"HCP_data.csv"))
n_subj=HCP_data.shape[0]

HCP_data=HCP_data.to_numpy()
HCP_data=HCP_data[:,0:170] # Last features will not be used as they are cortical thickness but from another ROI map


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


# Best Model
# [60, 5e-05, 0.0005, [100], 2000, 32]
best_hyperparameters={'model__h_dim': [[100]], 'model__z_dim': [60],
                      'model__learnRate': [0.00005], 'batch_size': [32],
                      'epochs': [2000], 'model__kernel_regul':[0.0005]}

best_hyper=ParameterGrid(best_hyperparameters)[0]

#cb=callbacks=[EarlyStopping(monitor='val_mse', patience=100,restore_best_weights=True), History()] #callbacks=[cb]
model=KerasRegressor(model=build_model,random_state=123)

#I set the model with parameters from param_comb
model=model.set_params(**best_hyper)

# LOO-CV: Extract RE of all HCP subjects

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
HCP_reconstruction=pd.DataFrame(np.zeros((HCP_data.shape[0], HCP_data.shape[1])))
HCP_RE=pd.DataFrame(np.zeros((HCP_data.shape[0], HCP_data.shape[1])))
main_folder="XXXX"

for i, (train_index, test_index) in enumerate(loo.split(HCP_data)):
    
    print('Iteration {}, HCP subject ID {}'.format(i, test_index))
    
    X_train_fold, X_one_out = HCP_data[train_index], HCP_data[test_index]
    
    covars_train_fold=copy.deepcopy(HCP_covars.iloc[train_index,:]).reset_index()
    
    X_train_dict={'data': X_train_fold , 'covariates': covars_train_fold}
      
    covars_one_out=copy.deepcopy(HCP_covars.iloc[test_index,:]).reset_index()
    X_oneout_dict={'data': X_one_out , 'covariates': covars_one_out}
      
 
    # Correct Biological Covariates in Data 
      
    # Fit_transform method in training set
    X_train_corr=pipeline_biocov_reg.fit_transform(X_train_dict)
    
    # Transforming test subj
    X_oneout_corr=pipeline_biocov_reg.transform(X_oneout_dict)
    
    # Train the model
    history=model.fit(X_train_corr, X_train_corr, shuffle=True)

      
    
    # Test the model
    X_oneout_reconstruction = model.predict(X_oneout_corr)
    
    HCP_reconstruction.iloc[test_index,:]=X_oneout_reconstruction

    #Calculate Reconstruction Error
    X_RE=copy.copy(np.square(X_oneout_reconstruction-X_oneout_corr))
    
    HCP_RE.iloc[test_index,:]=X_RE


    # Save deviations maps 
    HCP_RE_path=  'HCP_RE_' + str(test_index) + '.csv'
    path_save_results=os.path.join(main_folder,"HCP_reconstruction_error", HCP_RE_path)
    with open(path_save_results, 'w', encoding = 'utf-8-sig') as f:
      pd.DataFrame(X_RE).to_csv(f)
      
# Save HCP_Reconstruction dataset

path_save_results=os.path.join(main_folder,"HCP_reconstruction_error",'HCP_reconstruction.csv')
with open(path_save_results, 'w', encoding = 'utf-8-sig') as f:
  HCP_reconstruction.to_csv(f)
  
# Save HCP_Reconstruction dataset
 
path_save_results=os.path.join(main_folder,"HCP_reconstruction_error",'HCP_RE.csv')
with open(path_save_results, 'w', encoding = 'utf-8-sig') as f:
  HCP_RE.to_csv(f)