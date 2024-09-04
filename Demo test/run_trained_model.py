# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:21:11 2024

@author: Inês Won
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import pickle
from numpy import genfromtxt
from tensorflow import keras

def run_model(data_path, model_path, HCP_statistics_path, save_results_folder):
    # Load data
    data = load_data(data_path)
    
    print('data shape {}'.format(data.shape))

    # Load model
    model = load_model(model_path)
    
    # Load HCP median and mad by feature
    
    mZ_estimates= load_normative_estimates(HCP_statistics_path)
    
    # Preprocess data
    #preprocessed_data = preprocess_data(data)

    # Load pretrained model
    #model = YourModel()

    # Run inference
    reconstructions = model.predict(data)
    save_results(reconstructions, save_results_folder, 'reconstructions.pkl')

    # Calculate reconstruction error (not averaged)
    # In this way the averaging can be performed aposteriori by group (need for 
    # specific group IDs) depending on data and application
    squared_error = np.square(reconstructions-data) 
    save_results(squared_error, save_results_folder, 'squared_error.pkl')
 
    # Calculate mean reconstruction error: subjects MDS
    mean_deviation_scores = squared_error.mean(axis=1)
    save_results(mean_deviation_scores, save_results_folder, 'subjects_MDS.pkl')
    
    # Calculate mZ scores
    mZ_scores = ( squared_error - mZ_estimates['feat_median'] ) / mZ_estimates['feat_mad']
    save_results(mZ_scores, save_results_folder, 'mZ_scores.pkl')
    
def load_data(data_path):
    # Implement data loading logic here
    
    data = np.genfromtxt(data_path, delimiter=',')[1:,1:]
    
    return data

def load_model(model_path):
    
    # Open the pickle file and load the dictionary
    #with open(model_path, 'rb') as f:
    #    model = pickle.load(f)
        
    model=tf.keras.models.load_model(model_path)
    
    return model

def load_normative_estimates(estimates_path):
    
    # Open the pickle file and load the dictionary
    with open(estimates_path, 'rb') as f:
        mZ_estimates = pickle.load(f)
    
    return mZ_estimates
    
def save_results(results, save_results_folder, filename):
    # Create the save_results_folder if it doesn't exist
    os.makedirs(save_results_folder, exist_ok=True)
    
    # Define the full path for the results file
    results_file_path = os.path.join(save_results_folder, filename)
    
    # Save results to the specified file
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f) 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pretrained model on data")
    parser.add_argument("data_preprocessed", help="Path to the preprocessed .csv data file: shape (n_subjects X 170 features)")
    parser.add_argument("trained_model", help="Path to the trained model .h5 file")
    parser.add_argument("HCP_statistics_path", help="Path to estimated HCP statistics")
    parser.add_argument("save_results_folder", help="Folder to save the  results")
    args = parser.parse_args()

    # Run the model with the provided arguments
    run_model(args.data_preprocessed, args.trained_model, args.HCP_statistics_path, args.save_results_folder)
