# Normative_AE

# Folders information:

# Test the model with given examplar data:
In Demo_test all the instructions are given to run a demo. 
Download the folder and follow the instructions:
```python
python run_trained_model.py exemplar_HCP_corr_data/demo_HCP_data.csv trained_normative_autoencoder_model_2.h5 mZ_HCP_estimates.pkl <YOURresuls_folder_name>
```
# Test the model with new data:
1. MRI_preprocessing_matlab folder: includes all files for reproducing the MRI preprocessing in Matlab with SPM/CAT12;
   
2. Run the Confounder Removal Pipeline - ComBat harmonization and adjusting for biological covariates age and sex:
   
    2.1 HCP doesn't release age publicly, therefore to run the confounder_removal pipeline as in "Analysis Scripts -> Confounder_removal_application.ipynb" with new data the researcher must:
   
       a. Apply for access to Restricted Data https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage ;
   
       b. Download the HCP age information from the website (only for d_option1 : and match the subjects with the IDs provided file "HCP_randomized_order_subject_IDs");
   
       c. Follow the instructions to fit the ComBat model using the new data and the HCP data as reference ( general instructions can be found in "deconfounders_pyClasse -> Example_Application.ipynb" or checking 
          the specific application on "Analysis Scripts -> Confounder_removal_application.ipynb").
   
       d_option1. Run the new corrected test data with the already trained normative model available in "Demo_test -> trained_normative_autoencoder_model_2.h5" by following the instructions;

       d_option2. Train from scratch the model available in "AE_original_model.ipynb";

3. Check the bootstrap_script.ipynb to run the model with N bootstrap training sample iterations and then the process_bootstrap_results.ipynb;
   
4. Check the notebooks Heterogeinity_MEVs_analysis.ipnynb and Personalized_analysis_modified_z_scores.ipynb for the other analysis 
   
