# Normative Autoencoder
## *A generalizable normative deep autoencoder for brain morphological anomaly detection: application to the multi-site StratiBip dataset on bipolar disorder in an external validation framework.*

An autoencoder model was trained with structural MRI-based features (cortical thickness, gray and white matter volumes) of 1109 healthy young adults from the public 3T Human Connectome Project dataset[1] https://www.humanconnectome.org/study/hcp-young-adult and tested with an age range matched subsample of the multi-site StratiBip [2] private dataset composed of healthy and subjects affected with bipolar disorder. 

We embedded a confounding removal step in the analysis pipeline that integrated the training sample with the external test set. This step included an M-ComBat harmonization [3,4] and linearly regressing out biological covariates of no interest, age and sex [5,6].

After adjusting for confounding variables (site, age and sex) the HCP data was used to train the normative autoencoder model and the StratiBip to test the model by extracting individual and group-level deviation metrics. The latter included: 
   1. Identification of significantly deviating ROI in the BD group;
   2. Computing deviating heterogeinity patterns
   3. Computing extreme deviations for each feature by group and by subject
   4. Computing individual deviating maps employing the modified z-scores and analyzing group spatial overlap

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

# Cite this work 

# Main References
[1]  David C. Van Essen, Stephen M. Smith, Deanna M. Barch, Timothy E.J. Behrens, Essa Yacoub, Kamil Ugurbil, for the WU-Minn HCP Consortium. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage 80(2013):62-79. 

[2] Maggioni E, Crespo-Facorro B, Nenadic I, Benedetti F, Gaser C, Sauer H, Roiz-Santiañez R, Poletti S, Marinelli V, Bellani M, Perlini C, Ruggeri M, Altamura AC, Diwadkar VA, Brambilla P; ENPACT group. Common and distinct structural features of schizophrenia and bipolar disorder: The European Network on Psychosis, Affective disorders and Cognitive Trajectory (ENPACT) study. PLoS One. 2017 Nov 14;12(11):e0188000. doi: 10.1371/journal.pone.0188000. PMID: 29136642; PMCID: PMC5685634.

[3] Stein, C. K. et al. Removing batch effects from purified plasma cell gene expression microarrays with modified ComBat. BMC Bioinformatics 16, 1–9 (2015).

[4] Lim, C. H. et al. Development and External Validation of 18F-FDG PET-Based Radiomic Model for Predicting Pathologic Complete Response after Neoadjuvant Chemotherapy in Breast Cancer. Cancers (Basel). 15, (2023).

[5] Snoek, L., Miletić, S. & Scholte, H. S. How to control for confounds in decoding analyses of neuroimaging data. Neuroimage 184, 741–760 (2019).

[6] Manduchi, E., Fu, W., Romano, J. D., Ruberto, S. & Moore, J. H. Embedding covariate adjustments in tree-based automated machine learning for biomedical big data analyses. BMC Bioinformatics 21, 1–13 (2020).
