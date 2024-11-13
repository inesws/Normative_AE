# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:52:26 2023

@author: Utente
"""
""" Hyperparameter tunning """

#import model 
#import LR class
import random

# We need to install this for functioning of GridSearch
#!pip install scikeras
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.callbacks import EarlyStopping, History

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import ParameterGrid
from keras.callbacks import EarlyStopping, History
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold	
from sklearn.metrics import mean_squared_error
import time

#Datasets and system
import numpy as np                # Management of arrays
import os                         # System utils
import pandas as pd
from pathlib import Path          # path and file utils
from scipy.io import loadmat      # Required to load .mat files
import h5py #For creating new dataset files
import sklearn 
from scipy.stats import loguniform
import copy

"""
The model is defined with a function that returns the model compiles. 
The hyperparameters to be tunned should be passed as arguments to the function. 
The model is wrapped with KerasRegressor or KerasClassifiers 

"""  
def hyperparameter_tunning(X_train, covars_train,preprocessing_pipeline, random_search, model, hyperparameter_space,path_save, verbose ):
    
    start = time.time()
    n_feat=X_train.shape[1]
    n_subj=X_train.shape[0]
    
    #Best parameters based on training R.E

    best_param_RE=[]
    best_score_RE=10000 # to initialize parameter
    
    #Best parameters based on discrimination between HC test fold and BD validation set 
    best_param_AUC=[]
    best_score_AUC=0.5 # to initialize AUC
    X_corr=pd.DataFrame(np.zeros((n_subj,n_feat)))
    

    cb=callbacks=[EarlyStopping(monitor='val_mse', patience=100,restore_best_weights=True), History()]
    
    
    AE=KerasRegressor(model=model, callbacks=[cb], epochs=1000, batch_size=65,random_state=random_search['seed']) #using MSE as Loss build_fn
    
    #AE=KerasClassifier(build_fn=build_model, callbacks=[cb], random_state=0, epochs=2000) #using binary_crossentropy as Loss
    #print(AE.get_params().keys())


    grid = list(range(0, len(ParameterGrid(hyperparameter_space))))
    
    skf = KFold(n_splits=10, random_state=0, shuffle=True) #Stratified
    

    # Random Search Definition
    max_iter=random_search['max_iter']
    seed=random_search['seed']
    curr_iter=0

    # Save results file
    hyper_results=pd.DataFrame(np.zeros((max_iter,5)), columns=['seed','c_id','Parameter_Comb','RE_val','RE_best_score']) #'AUC','AUC_best_score'
    hyper_results.loc[:,('seed')]=seed


    random.seed(a=seed)    
    for c in random.sample(grid,max_iter): # iterate in the grid of hyperparameter combinations 
      cv_10=[]
      #auc_cv=[]
      param_comb=ParameterGrid(hyperparameter_space)[c] # The parameter combination number c
      
      if verbose>1:
          print('Parameter Combination ID: {}'.format(c))
          print( 'Iteration nº {} from {}'.format(curr_iter, max_iter))
      
      for train_index, kfold_index in skf.split(X_train,X_train):
          
          
          # Perform 10-fold split # batch_hc_train
          if verbose>1:
              print('CV-Fold nº {} from 10'.format(len(cv_10)))
        
          X_train_fold, X_test_fold = X_train[train_index], X_train[kfold_index] # I get the division of the training set into Train and Test
        
          covars_train_fold=copy.deepcopy(covars_train.iloc[train_index,:]).reset_index()
          X_train_dict={'data': X_train_fold , 'covariates': covars_train_fold}
        
          covars_test_fold=copy.deepcopy(covars_train.iloc[kfold_index,:]).reset_index()
          X_test_dict={'data': X_test_fold , 'covariates': covars_test_fold}
        
          #covars_bd_val_copy=copy.deepcopy(covars_bd_val)
        
        
          # Correct Data 
        
          if verbose>=2:
              print('Data Processing Pipile')
        
          # Fit_transform method in training set
          X_train_corr=preprocessing_pipeline.fit_transform(X_train_dict)
          
        
          # Transforming test fold
          X_test_corr=preprocessing_pipeline.transform(X_test_dict)
        
        
          #I set the model with parameters from param_comb
          model=AE.set_params(**param_comb) 
        
          #I fit evalutating the model on kth-fold -X_test_corr
          history=model.fit(X_train_corr,X_train_corr, shuffle=True, validation_data=(X_test_corr,X_test_corr))
           
          val_mse=history.history_["val_mse"][-1] #last val_mes which is the same as doing: np.square(X_test_corr - HC_val_hat).mean(axis=1).mean(9)
          #val_loss=history.history_["val_loss"][-1] #last val_loss
        
          #I save the validation dat mse for the k-th fold
          cv_10.append(val_mse) 
    
        # Reconstruct data
        #HC_val_hat = model.predict(X_test_corr)
    
      #Val Reconstruction Error
      avg_mse_cv=sum(cv_10) / len(cv_10)
      
      #avg_auc_cv=sum(auc_cv)/len(auc_cv)
    
      hyper_results.loc[curr_iter,'c_id']=c
      hyper_results.loc[curr_iter,'Parameter_Comb']= str(list(param_comb.values()))
      hyper_results.loc[curr_iter,'RE_val']=avg_mse_cv
      #hyper_results.loc[curr_iter,'AUC']=avg_auc_cv
      hyper_results.loc[curr_iter,'RE_best_score']=0
      #hyper_results.loc[curr_iter,'AUC_best_score']=0
    
      #avg_ov=sum(overfit) / len(overfit)
      if avg_mse_cv < best_score_RE: #If the average is lower (best mse is lower mse) then the prior best_score found, I save the new best_score and the correspondent hyperparameters
        best_c=c
        best_score_RE=avg_mse_cv
        best_param_RE=param_comb 
        hyper_results.loc[curr_iter,'RE_best_score']=curr_iter+1
    
      curr_iter+=1
      
      #Save Results
      with open(path_save, 'w', encoding = 'utf-8-sig') as f:
        hyper_results.to_csv(f)
    
    end = time.time()
    diff = end - start
    print('Execution time of Random Search (in minutes): {}'.format(diff/60))

    return hyper_results