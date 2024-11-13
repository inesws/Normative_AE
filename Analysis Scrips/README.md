Pipeline:

  Data Correction
1. Confounder_removal_application.ipynb : corrected data for confounders;

   Hyperparameter tunning: find best model architecture
3. autoencoder_model_definition.py : defined model for hyperparameter tunning;
4. hyperparameter_tunning.py : defined an hyperparameter function;
5. Train_model.py : applied the hyperparameter search;

   Group-level analysis
6. bootstrap_script : with best hyperparameters runned a bootstrapped analysis to evaluate model stability ;
7. process_bootstrap_results.ipynb: calculated 95% CI etc from the bootstrap analysis;
8. Heteogeinity_MEVs_analysis.ipynb : analyzed the mean deviation scores and mean extreme deviations patterns ;

   
   mZ scores
9. LOOCV_extract_HCP_RE.py: run a leave-one-out CV to extract the unbiased training set RE with the best model -> used for the mZ scores analysis bellow 
10. AE_original_model.ipynb: model analysis without bootstrap -> used for the mZ scores analysis bellow
11. Personalized_analysis_modified_zscores.ipynb : calculated the mZ scores and analysed the spatial overlap of individual brain deviating maps
