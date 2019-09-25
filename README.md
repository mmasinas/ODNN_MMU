# ODNN
This repository contains the code and example datasets for the paper:  

"Exploring endocytic compartment morphology with systematic genetics and single cell image analysis"

Mojca Mattiazzi Usaj, Nil Sahin, Helena Friesen, Carles Pons, Matej Usaj, Myra Paz Masinas,
Ermira Shuteriqi, Aleksei Shkurin, Patrick Aloy, Quaid Morris, Charles Boone, and Brenda J. Andrews

#### Mutant Phenotype Classification and Penetrance Calculation
This script learns and predicts mutant phenotypes from genetic or chemical perturbation screens.
_CellProfiler*_ is used for extracting features but any other software or custom features can be used.
The plates are normalized with respect to normal (wild-type, WT) cells each plate.
A fully neural network with 2 hidden layers is trained and used for single cell phenotype classification.
The predictions are combined into well and strain for phenotype percentages
and penetrance as percent cells with combined mutant phenotypes.

*Carpenter AE, et al. CellProfiler: image analysis software for identifying and quantifying cell phenotypes.
_Genome Biology_ 7:R100. PMID: 17076895 (2006).


## Packages

Python 3+: http://www.python.org/getit/
   
Tensorflow 1.13+: https://www.tensorflow.org/install
   
Keras 2.2+: https://keras.io/#installation

You can use the following command to install all the required packages:

    conda create --name <env> --file requirements.txt python=3.6.2 anaconda


## Requirements
* This analysis is made on an array format. Plate-Row-Column information should be input.
* At least one well identifier is required. This can be strain or condition information.
* Every cell should have a unique ID number (cell_ID). This can be generated at any software to process sheets.
* Positive and negative controls should have all well identifiers. 'negative' phenotype is required in the control file.



## Full Datasets

The datasets are too large to store in the repository.

The datasets are available at:
<http://data_link.com>


## Running ODNN

To run ODNN, please use Option Parser, for example:

    python ODNN.py --input-files input/screen1_plates.txt
    --features input/screen1_feature_set.txt
    --mapping-sheet input/screens_mapping_sheet.csv
    --identifier strain
    --control input/screen1_controls.csv

### Input Options

**--input-files** (-i): A list of plates to be analyzed.
The paths of the plates should contain a .csv file of single cell features and a column to map the location and well identifiers.
Each row of these .csv files is a single cell, and they should have a unique ID number (cell_ID). This can be generated at any software to process sheets.  
Example file at _input/screen1_plates.txt_

**--input-data** (-d): A complete screen data scaled for all features and each row has well identifiers.
Please use this if you don't want the script to scale data and/or map rows with well identifier information from the mapping sheet option.  
Example file at _input/screen1_data_scaled.csv_

**--features** (-f): A list of features to be used for the analysis.
There are features such as filename and location from CellProfiler that should not be included in the feature data for analysis.  
Example file at _input/screen1_feature_set.txt_

**--mapping-sheet** (-m): This sheet contains strain information for each well in each plate of the screen.
CellProfiler do not have an input mapping sheet option so each row of CellProfiler output file contains an image name and feature names that can be mapped with a mapping sheet.  
Example file at _input/screens_mapping_sheet.csv_

**--control** (-c): This sheet contains positive and negative control strains.
Positive controls are the mutant phenotype classes known prior, negative controls are the normal (WT, unperturbed) strains.  
Example file at _input/screen1_controls.csv_  

**--control** (-c): If you would like to assess the performance of penetrance calculation (total % mutant cells in a strain), please put a column named "Bin" and label each control with your manual assessment accordingly:  
Bin-1: 80-100% penetrance  
Bin-2: 60-80% penetrance  
Bin-3: 40-60% penetrance  
Bin-4: 20-40% penetrance  
Bin-0: 0-20% penetrance - negative controls  
Example file at _input/screen1_controls_penetrance_bins.csv_

**--pos-control-cell** (-x): The labeled set can also consist of individually labeled cells.
The data for these single cells will be obtained from the input files.
This file has a cell_ID column and all of the columns in the --control file.  
Example file at _input/screen1_positive_controls_singlecell.csv_

**--pos-control-celldata** (-y): The labeled set can also consist of individually labeled cells and scaled data for each cell.
This is especially useful if you would like to use labeled cells from another screen to make classification on a screen you don't have enough prior information.
This file has a cell_ID column, all of the columns in the --control file, and feature data.  
Example file at _input/screen1_positive_controls_singlecell_data.csv_

**--probability** (-p): The minimum probability required for the phenotype class with maximum probability from the softmax layer of the neural network to make a prediction.
If this minimum probability is not obtained, the prediction will not be made and that cell will be predicted as having **none** class. Default is 0, so there will be a phenotype prediction, even if the maximum class has a low probability.

**--identifier** (-u): The column name from mapping sheet to be used as the unique well identifier. This can be for strains or conditions.
There could be different alleles with the same ORF so that a unique well identifier is needed for combining the results.


### Hyper-parameters for NN

There are a lot of hyper-parameters to run a neural network, creating options for each would not be feasible.
Please change these parameters in the code _ODNN.py_.

    # Neural network hyper-parameters
    param = {'hidden_units' : [54, 18],
             'percent_to_test': 0.2,
             'percent_to_valid' : 0.2,
             'batch_size': 100,
             'k_fold_cv': 5,
             'learning_rate' : 0.01,
             'decay' : 1e-6,
             'momentum' : 0.9,
             'nesterov' : True,
             'num_epochs' : 50,
             'runs': 10}
             
**hidden_units**: Number of hidden units on two hidden layers in a list.
The neural_network() function needs to be changed if more hidden layers are needed. _[int, int]_

**percent_to_test**: Percentage of labeled dataset to be set aside for test set. _float_ [0, 1] 

**percent_to_validate**: Percentage of training set to be used for validation set during training. _float_ [0, 1] 

**batch_size**: Number of samples per gradient update. _int_ [32, len(training_set)]

**k_fold_cv**: Number of folds used in k-fold cross-validation during training. _int_

**learning_rate**: Learning rate used in Stochastic Gradient Descent (SGD) optimizer. _float_ >=0

**decay**: Learning rate decay over each update in SGD. _float_ >=0

**momentum**: Parameter that accelerates SGD in the relevant direction and dampens oscillations. _float_ >=0

**nesterov**: Whether to apply Nesterov momentum. _boolean_

**num_epoch**: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. _int_

**runs**: Number of independent training. The average predictions of these runs are the final predictions on single cells. _int_


### Output Files

All output files are saved in the folder _ODNN_results_ in the location that the script is run.

**Classification_results_strain.csv**: Penetrance calculation and percentage of cells in each phenotype class for each strain/condition.
If multiple wells have the same strain/condition, they are combined in this result. P-value is also calculated under the null negative distribution.  

**Classification_results_well.csv**: Penetrance calculation and percentage of cells in each phenotype class for each well.
P-value is also calculated under the null negative distribution.  

**Confusion_matrix.png**: Performance of NN classification on labeled data from cross-validation.  

**Data_scaled.csv**: For each cell and feature, the features are scaled with respect to negative cells for zero mean and unit variance. 
The scaled features and single cell strain and location identifiers are saved in this file.  

**Single_cell_phenotype_predictions.csv**: Phenotype predictions for single cells in the complete screen. 
The average probability for each phenotype is displayed. If the maximum probability is less than the threshold, the prediction will be _none_.  

**Performance_penetrance_bins.csv**: If the control files have manual penetrance assessment bins, the penetrance calculated and predicted bins are displayed in this file.  

**Performance_ROC_PR.csv**: Precision, true positive rate (TPR, Recall) and false positive rate (FPR) are calculated by changing the penetrance cutoff based on negative control percentiles on penetrance values.
These values are calculated by the performance in capturing positive controls and falsely calling negative controls as positives.  

**Positive_controls_data.csv**: Labeled data used to train NN is saved. Each row is a single cell with strain and location information, scaled features, and labeled phenotype.  

**Single_cell_accuracies.csv**: The accuracy for each labeled cell is calculated from phenotype predictions for all NN run and labeled phenotype. 
The phenotype predictions of each run and the average probability for each phenotype class is displayed.  
