Pipeline:

1. Confounder_removal_application.ipynb : corrected data for confounders;
2. autoencoder_model_definition.py : defined model for hyperparameter tunning;
3. hyperparameter_tunning.py : defined an hyperparameter function;
4. Train_model.py : applied the hyperparameter search;
5. bootstrap_script : with best hyperparameters runned a bootstrapped analysis to evaluate model stability ;
6. process_bootstrap_results.ipynb: calculated 95% CI etc from the bootstrap analysis;
7. LOOCV_extract_HCP_RE.py: run a leave-one-out CV to extract the unbiased training set RE with the best model ;
8. Heteogeinity_MEVs_analysis.ipynb : analyzed the mean deviation scores and mean extreme deviations patterns ;
9. Personalized_analysis_modified_zscores.ipynb : calculated the mZ scores and analysed the spatial overlap of individual brain deviating maps
