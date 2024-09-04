# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:04:23 2023

@author: Utente
"""


#--------Bootstrap script----------

from sklearn.model_selection import LeaveOneOut
from keras.engine.training import Model
from tensorflow.keras.models import save_model
import pickle
#import sys
#sys.path.insert(0,"***\\Functions_Classes")
#import confounder_correction_classes
#from confounder_correction_classes import ComBatHarmonization,StandardScalerDict, BiocovariatesRegression

#Datasets and system
import numpy as np                # Management of arrays
import os                         # System utils
import pandas as pd
from pathlib import Path          # path and file utils
from scipy.io import loadmat      # Required to load .mat files
import h5py #For creating new dataset files
import sklearn
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Bootstrap
from sklearn.utils import resample

# Model
import sys
sys.path.insert(0,"***\\AE_normative\\")
from basic_model import build_model

!pip install scikeras
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, History
from sklearn.model_selection import ParameterGrid

# Load and Prepare Data
main_folder= "C:***\\AE_normative\\"
data_path="C:\\***\\combat_modmean_lr_betas\\"

# HCP
HCP_covars=pd.read_csv(os.path.join(data_path, "HCP_covars_corr.csv"))
HCP_data=pd.read_csv(os.path.join(data_path, "HCP_data_corr.csv"))
n_HCP=HCP_data.shape[0]

# ***
Strat_covars=pd.read_csv(os.path.join(data_path, "***_YA_covar_corr_withglobals.csv"))
Strat_data=pd.read_csv(os.path.join(data_path, "***_YA_data_corr_withglobals.csv"))
n_Strat=Strat_data.shape[0]

hc_index=Strat_covars[Strat_covars['Diagnosis']==0].index
bd_index=Strat_covars[Strat_covars['Diagnosis']!=0].index


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

# Bootstrap: x1000 with replacement

n_iter=1000
plot_loss=[]

for i in range(0,n_iter):

  hcp_index=np.arange(0,n_HCP)
  bootstrap_id= resample(hcp_index, replace=True, random_state=i)
  print('Iteration n {} of 1000'.format(i))
  print(bootstrap_id)
  
  bootstrap_sample=copy.deepcopy(HCP_data.iloc[bootstrap_id,:])
  bootstrap_covars=copy.deepcopy(HCP_covars.iloc[bootstrap_id,:])#.reset_index()

  history=model.fit(bootstrap_sample, bootstrap_sample, shuffle=True, validation_split=0.1)

  model_name='model_' + str(i) + '.h5'
  model.model_.save(os.path.join(main_folder,"Bootstrap_Results\\combat_modmean_lr_betas\\bootstrap_models\\", model_name))
 
  # Generate a colormap with n_iter colors
  cmap = mcolors.ListedColormap(np.random.rand(n_iter,3))
  plot_loss.append([history.history_["loss"],history.history_["val_loss"]])
  plt.plot(history.history_["loss"], label="Training Loss", linestyle='-', color=cmap(i))
  plt.plot(history.history_["val_loss"], label="Validation Loss", linestyle='--', color=cmap(i))

  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  #plt.legend()

  # Test the model
  Strat_reconstructions = model.predict(Strat_data)

  #Calculate Deviation Matrix
  strat_dev_map=copy.copy(np.square(Strat_reconstructions-Strat_data))
  strat_mean_dev_map=strat_dev_map.mean(axis=1)
  #strat_mse = tf.keras.losses.mse(Strat_reconstructions, Strat_data)

  HC_strat_dev_map=strat_dev_map.iloc[hc_index,:]
  BD_strat_dev_map=strat_dev_map.iloc[bd_index,:]
  #HC_strat_mean_dev=strat_mean_dev_map[hc_index]
  #BD_strat_mean_dev=strat_mean_dev_map[bd_index]

  #Save Results

  # Save deviations maps 
  hc_dm_path=  'HC_strat_dev_map_boot_' + str(i) + '_.csv'
  bd_dm_path=  'BD_strat_dev_map_boot_' + str(i) + '_.csv'
  #hc_mdm_path=  'HC_strat_mean_dev_map_boot_' + str(i) + '_.csv'
  #bd_mdm_path=  'BD_strat_mean_dev_map_boot_' + str(i) + '_.csv'
  path_save_results=os.path.join(main_folder,"Bootstrap_Results\\combat_modmean_lr_betas\\bootstrap_deviation_maps\\", hc_dm_path)
  with open(path_save_results, 'w', encoding = 'utf-8-sig') as f:
    HC_strat_dev_map.to_csv(f)

  path_save_results=os.path.join(main_folder,"Bootstrap_Results\\combat_modmean_lr_betas\\bootstrap_deviation_maps\\", bd_dm_path)
  with open(path_save_results, 'w', encoding = 'utf-8-sig') as f:
    BD_strat_dev_map.to_csv(f)
  
  # Save Loss Plot Figure
  with open(os.path.join(main_folder,"Bootstrap_Results\\combat_modmean_lr_betas\\bootstrap_models\\boot_losses"), "wb") as fp:
    pickle.dump(plot_loss, fp)


# Generate a colormap with n_iter colors
#cmap = mcolors.ListedColormap(np.random.rand(1000,3))

# Load Loss Plot Figure
with open(os.path.join(main_folder,"Bootstrap_Results\\combat_modmean_lr_betas\\bootstrap_models\\boot_losses"), "rb") as fp:
  plot_loss = pickle.load(fp)

plt.figure()
for boot, (loss_train, loss_val) in enumerate(plot_loss):
    plt.plot(loss_train,label="Training Loss", linestyle='-', color=cmap(boot))
    plt.plot(loss_val,label="Validation Loss", linestyle='--', color=cmap(boot))
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.legend()




