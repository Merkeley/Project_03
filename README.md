# Predicting Telecom Customer Churn

This directory is organized as follows

Project_03  -|<br>
             |- notes - background information and presentations<br>
             |- images - images used in presentations<br>
             |- Notebooks - Jupyter notebooks created for this project<br>
             |- src - python scripts created for this project<br>
             |- data - primary and intermediate data files used in this project<br>

## Data
The data for this project was obtained from Kaggle at this location:
https://www.kaggle.com/blastchar/telco-customer-churn

The data is in the form of a csv file.

## Notebooks
There are many notebooks for this project.  The notebooks are broken up by task.  The workflow for the project uses the notebooks in the following order

1. **Telecom_Data_to_SQL** opens the data file and stores the data in a postgres database on an AWS instance.
2. **Telecom_Data_Clean_Explore** runs a script to clean the data.  A script is used for consistency and repeatability.
3. **Model_search, Model_hybrid_search** are notebooks the implement grid search cross validation to find the best model of each type.
4. **Model_final, Model_final_hybrid** are notebooks that run the final models of each type and plot the results.
5. **Model_Evaluation** is the notebook that uses the best model of all those tested and applies it to the test data set that was held out at the beginning.

## Src
There are two script files that were developed for this project.

**telecom_dat_clean** implements two scripts that are used to clean the data frames created from the csv files.

**my_eval_tools** contains several functions that I used during the model evaluation portion of this project.


