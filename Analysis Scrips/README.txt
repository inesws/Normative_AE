These scripts contained all the analysis and results as reported in the manuscript. 
Due to the inability of sharing any data, the scripts are intended for visualization only. 

AE_model_definition.py : the autoencoder model;

AE_original_model : implementing the original model in all data (without bootstrap analysis);

Confounder_removal_application.ipynb : the CR pipeline application to the datasets;

confounder_correction_classes_combat_and_lr : the ComBat and Biocovariates Linear Regression python classes developed for an easy application.

neurocombat_fun_modified : neurocombat modified to transform a new set including when estimations were performed with M-ComBat variation.

bootstrap_script.py : the scripted that runned the model with 1000 different boostrap samples from the HCP-YA dataset.

process_bootstrap_results.ipynb : the analysis where the bootstrap 95% CI was extracted and evaluated.

Personalized_analysis_modified_zscores.ipynb : all the analysis related to the personalized individual brain deviating maps. 

Heterogeinity_MEVs_analysis.ipynb : the heterogeinity and extreme values group-level analysis performed on the original model (not bootstraped models).

*Disclaimer: 

Because Age is not release by the HCP-YA public dataset it is not possible to share the fitted confounder removal pipeline to apply on new data. 
Both M-ComBat and LR models were fitted on the normative HCP-YA training set using Age as covariate. 

To harmonize and standardize new data with the normative training set a researcher must individually ask for accessing sensible data (such as AGE) from the HCP dataset. 
Then it's possible to harmonize new data with the reference by replicating the code as in "Confounder_removal_application.ipynb".