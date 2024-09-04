*File information:*
. examplar_HCP_corr_data: Example data to demonstrate the functioning of the normative model. 
includes 10 subjects randomly selected from the HCP training set, already preprocessed 
and passed through the confounder removal pipeline.

. mZ_HCP_estimates.pkl: contains the HCP feature-wise RE median and median absolute deviation (MAD) necessary to calculate the mZ scores for new data. dictionary structured like {'median': np.array(1,170) ; 'mad': np.array(1,170)} saved in a pickle file.

. trained_normative_autoencoder_model_2.h5: trained normative model saved in h5 file

*Instruction to try normative model with example data with a python interpreter:*
Be sure to run the code with the following env versions:

python version 3.12.3

conda version 24.5.0

(optional) spyder version 5.4.3

Download folder

If needed install anaconda/conda version (24.5.0)

Create a conda environment with the requirements.txt

Write in the Windows command line:

a. conda create --name normative_model python=3.12.3

b. conda activate normative_model

(go to folder path by doing cd \path\to\test_model_demo)

c. pip install -r requirements.txt

Run the following command in the command line:
python run_trained_model.py exemplar_HCP_corr_data/demo_HCP_data.csv trained_normative_autoencoder_model_2.h5 mZ_HCP_estimates.pkl <YOURresuls_folder_name>

*Instruction to try normative model with example data with the .exe file :*

Run the following command in the command line:
un_trained_model.exe exemplar_HCP_corr_data/demo_HCP_data.csv trained_normative_autoencoder_model_2.h5 mZ_HCP_estimates.pkl <YOURresuls_folder_name>