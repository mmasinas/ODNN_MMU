# Copyright (c) 2018, Mojca Mattiazzi Usaj & Nil Sahin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Required packages

import matplotlib
matplotlib.use('Agg')
import sys
import os
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, losses
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_input_files(plate_list, input_data, features, map_file, location):
    """ Read input files for plates, CP features, mapping sheet.
        Return in lists and dataframe.

        Args:
            filename:       File of plate list to be analyzed
            input_data:     Input data, scaled and filled with identifiers
            features:       File of CP feature list to be used in the analysis
            map_file:       Map gene names from plate-row-column info.
            location:       Column names for plate-row-column info.

        Return:
            plates:         Plates to be analyzed
            feature_set:    Features to be analyzed
            mapping_sheet:  Dataframe to map strain/condition information from arrayed plates
            map_features:   Strain/condition identifier columns
        """

    # Read plate list
    plates = []
    if os.path.isfile(plate_list):
        f = open(plate_list, 'r')
        plates = list(filter(None, [x.strip() for x in f.readlines()]))
        f.close()
    elif not os.path.isfile(input_data):
        sys.exit('Please provide a list of plate paths to analyze, quitting...')

    # Read CP feature list
    if os.path.isfile(features):
        f = open(features, 'r')
        feature_set = list(filter(None, [x.strip() for x in f.readlines()]))
        f.close()
    else:
        sys.exit('Please provide a CP feature list, quitting...')

    # Read strain/condition information file
    if os.path.isfile(map_file):
        mapping_sheet = lower_column_names(pd.read_csv(map_file)).fillna('')
        map_features = mapping_sheet.columns.values.tolist()
        for i in location:
            map_features.remove(i)
    else:
        sys.exit('Please provide a mapping sheet for strain/condition information, quitting...')

    return plates, feature_set, mapping_sheet, map_features


def lower_column_names(df):
    """ Return pandas dataframe with lower column names.

        Args:
            df: Pandas DataFrame object

        Return:
            df: Pandas DataFrame object with lowercase column names
        """

    column_names = df.columns.values.tolist()
    column_names = [c.lower() for c in column_names]
    df.columns = column_names

    return df


def read_negative_controls_file(filename, identifier):
    """ Read controls file and return negative controls.

        Args:
            filename:       Controls file with positive and negative controls
            identifier:     Unique identifier for the strain/condition

        Return:
            neg_controls:   List of negative controls
            """

    if os.path.isfile(filename):
        neg_df = pd.read_csv(filename)
        neg_df = lower_column_names(neg_df)
        neg_df = neg_df[neg_df['phenotype'] == 'negative']
        neg_controls = neg_df[identifier].values.tolist()

    else:
        sys.exit('Please provide a controls file, quitting...')

    return neg_controls


def initialize_dictionary(identifiers):
    """ Return an empty dictionary with the following attributes.

        Args:
            identifiers:    Strain/condition identifier columns

        Return:
            df:             Dictionary of combined data
            dict_feat:      Dictionary features
            """

    df = {}
    for i in (identifiers + ['cell_id', 'column', 'row', 'plate', 'data', 'data_scaled', 'mask_neg']):
        df[i] = np.array([])

    dict_feat = list(df.keys())

    return df, dict_feat


def read_and_scale_plate(df, plate, neg, features, mapping_sheet, identifier, identifiers, dict_feat):
    """ Read, scale and add plate to the existing combined dictionary.

        Args:
            df:             Dictionary of combined data
            plate:          Plate name
            neg:            Negative controls for scaling
            features:       Features to be analyzed
            mapping_sheet:  strain information on plates
            identifier:     Unique identifier for the strain/condition
            identifiers:    Strain/condition identifier columns
            dict_feat:      Dictionary features

        Return:
            df:             Dictionary with new plate data appended
        """

    print('Reading: %s' % plate)

    # Read plate
    input_df = pd.read_csv(plate, header=0)
    input_df = lower_column_names(input_df)
    input_df['plate'] = plate
    plate_df = extract_plate_information(input_df, plate, features, mapping_sheet, neg, identifier, identifiers)

    # Scale plate wrt to negative control cells
    data_wt = plate_df['data'][plate_df['mask_neg'] == 1]
    plate_df['data_scaled'] = standard_scaler_fit_transform(data_wt, plate_df['data'])
    plate_df['data_scaled'][np.isnan(plate_df['data_scaled'])] = 0

    # Add to the existing dictionary of combined data
    df = append_data(df, plate_df, dict_feat)

    return df


def extract_plate_information(df, filename, features, mapping_sheet, neg, identifier, identifiers):
    """ Read a CP output for a plate and extract required information.

        Args:
            df:             CP output of a plate as a dataframe
            filename:       Filename of the plate
            features:       Features to be analyzed
            mapping_sheet:  Dataframe to map gene and allele names
            neg:            Negative controls for scaling
            identifier:     Unique identifier for the strain/condition
            identifiers:    Strain/condition identifier columns

        Return:
            plate_df:       Extracted plate information as a dictionary
        """

    # Extract plate locations and cell_IDs from CP output
    plate_df, _ = initialize_dictionary(identifiers)
    plate_df['cell_id'] = np.array(df['cell_id'], dtype='int32')
    plate_df['column'] = np.array(df['col'], dtype='int32')
    plate_df['row'] = np.array(df['row'], dtype='int32')
    plate_df['plate'] = np.array(df['plate'])

    # Map strain/condition information from arrayed format
    plate_name = filename.split('/')[-1].split('_')[0].replace('plate', '')
    ms_plate = mapping_sheet[mapping_sheet.plate == plate_name]

    # Initialize strain identifier lists
    identifier_lists = []
    for i in identifiers:
        identifier_lists.append([])

    # Add strain identifier information for each well
    for i in range(len(plate_df['column'])):
        map_df = ms_plate[(ms_plate['column'] == plate_df['column'][i]) & (ms_plate['row'] == plate_df['row'][i])]
        for j in range(len(identifier_lists)):
            identifier_lists[j].append(map_df[identifiers[j]].values[0])

    # Add strain identifier information to the plate dictionary data
    for i in range(len(identifiers)):
        plate_df[identifiers[i]] = np.array(identifier_lists[i])

    # Keep CP features needed for analysis
    plate_df['data'] = np.array(df[features], dtype='float64')

    # Find the negative control cells in the data
    plate_df['mask_neg'] = np.array([x in neg for x in plate_df[identifier]])

    return plate_df


def standard_scaler_fit_transform(data_fit, data_transform=np.array([])):
    """ Scale data into zero mean and unit variance for each column.

        Args:
            data_fit:       data to calculate mean and covariances
            data_transform: data to scale wrt data_fit

        Return:
            Scaled data
        """

    mean = np.nanmean(data_fit, axis=0)
    variance = np.nanstd(data_fit, axis=0)

    if len(data_transform)>0:
        return (data_transform - mean) / variance
    else:
        return (data_fit - mean) / variance


def save_data(df, features, identifiers, output):
    """ Save combined scale data for all plates.

        Args:
            df:             Dictionary of combined data
            features:       Features to be analyzed
            identifiers:    Strain/condition identifier columns
            output:         Output filename
        """

    # Save scaled data for all plates
    data_scaled_columns = ['cell_id', 'plate', 'row', 'column'] + identifiers + features
    data_scaled_output = df['cell_id'].reshape(-1, 1)
    for i in (['plate', 'row', 'column'] + identifiers):
        data_scaled_output = np.concatenate((data_scaled_output, df[i].reshape(-1, 1)), axis=1)
    data_scaled_output = np.concatenate((data_scaled_output, df['data_scaled']), axis=1)
    data_scaled_output_df = pd.DataFrame(data=data_scaled_output, columns=data_scaled_columns)
    data_scaled_output_df.to_csv(path_or_buf=output['DataScaled'], index=False)


def append_data(df, add_df, dict_feat):
    """ Append two dictionaries on selected features.

        Args:
            df:         Dictionary of combined data
            add_df:     Dictionary to be added
            dict_feat:  Dictionary features

        Return:
            df:         Dictionary of combined data
        """

    # Create a deep copy of features
    append_list = dict_feat[:]
    if len(df['data']) == 0:
        df['data'] = add_df['data']
        df['data_scaled'] = add_df['data_scaled']
        append_list.remove('data')
        append_list.remove('data_scaled')

    # Append new dictionary to the existing one
    for i in append_list:
        df[i] = np.append(df[i], add_df[i], axis=0)

    return df


def read_scaled_data(df, input_data, dict_feat, neg, identifier, features):
    """ Read an input file with scaled CP features and strain/condition identifier information.

        Args:
            df:             Initialized dictionary
            input_data:     Input data, scaled and filled with identifiers
            dict_feat:      Dictionary features
            neg:            Negative controls for scaling
            identifier:     Unique identifier for the strain/condition
            features:       Features to be analyzed

        Return:
            df:             Dictionary of combined data
        """

    print('Returning scaled data from %s' % input_data)

    # Read scaled data and save in a dataframe
    df_scaled = pd.read_csv(input_data)
    df_scaled = lower_column_names(df_scaled)
    exclude = ['data', 'data_scaled', 'mask_neg']
    for i in dict_feat:
        if i not in exclude:
            df[i] = np.array(df_scaled[i])

    # Use the original order of features
    df['data_scaled'] = np.array(df_scaled[features])
    df['data_scaled'][np.isnan(df['data_scaled'])] = 0
    df['mask_neg'] = np.array([x in neg for x in df[identifier]])

    return df


def prepare_phenotype_data(df, identifier, identifiers, features, pos_controls_files, output):
    """ Read and prepare training set from phenotype data with positive controls.

        Args:
            df:                     Dictionary of combined data
            identifier:             Unique identifier for the strain/condition
            identifiers:            Strain/condition identifier columns
            features:               Features to be analyzed
            pos_controls_files:     List of positive controls file
            output:                 Output filename

        Return:
            df:                     Dictionary with positive control information added
            phenotype_df:           Labeled set as a dataframe
            phenotype_classes:      List of phenotype classes
        """

    # Positive controls
    pos_controls_strain = pos_controls_files[0]
    pos_controls_cell = pos_controls_files[1]
    pos_controls_celldata = pos_controls_files[2]
    phenotype_classes = []

    # If training set with single cell data is available
    if pos_controls_celldata:
        print('\nPreparing phenotype data with transferred labeled set...')
        phenotype_df = pd.read_csv(pos_controls_celldata)
        phenotype_df = lower_column_names(phenotype_df)
        phenotype_classes = phenotype_df['phenotype'].unique()

    else:
        phenotypes = []
        # If training set with single cell IDs is available
        if pos_controls_cell:
            print('\nPreparing phenotype data with single cell labeled set...')
            pc = pd.read_csv(pos_controls_cell)
            pc = lower_column_names(pc)
            phenotype_classes = pc['phenotype'].unique()
            df['mask_pc'] = np.array([x in pc['cell_id'].values for x in df['cell_id']])
            phenotypes = [pc[pc['cell_id'] == x]['phenotype'].values[0] for x in df['cell_id'][df['mask_pc'] == 1]]

        # If training set with strain/condition is available, use all cells
        elif pos_controls_strain:
            print('\nPreparing phenotype data with strain labeled set...')
            pc = pd.read_csv(pos_controls_strain)
            pc = lower_column_names(pc)
            phenotype_classes = pc['phenotype'].unique()
            df['mask_pc'] = np.array([x in pc[identifier].values for x in df[identifier]])
            phenotypes = [pc[pc[identifier] == x]['phenotype'].values[0] for x in df[identifier][df['mask_pc'] == 1]]

        # Combine positive control data
        phenotype_data_columns = ['cell_id'] + identifiers + ['plate', 'row', 'column', 'phenotype'] + features
        phenotype_data = df['cell_id'][df['mask_pc'] == 1].reshape(-1, 1)
        for i in (identifiers + ['plate', 'row', 'column']):
            phenotype_data = np.concatenate((phenotype_data, df[i][df['mask_pc'] == 1].reshape(-1, 1)), axis=1)
        phenotype_data = np.concatenate((phenotype_data, np.array(phenotypes).reshape(-1,1),
                                         df['data_scaled'][df['mask_pc'] == 1]), axis=1)
        phenotype_df = pd.DataFrame(data=phenotype_data, columns=phenotype_data_columns)

    # Save positive control data
    phenotype_df = phenotype_df.fillna('').reset_index(drop=True)
    phenotype_df.to_csv(path_or_buf = output['PhenotypeData'], index=False)

    # Add None as a phenotype class
    phenotype_classes = np.append(phenotype_classes, ['none'])

    return df, phenotype_df, phenotype_classes


def split_labeled_set(pheno_df, features, k):
    """ Split labeled set into training and test set for k-fold cross-validation

        Args:
            pheno_df:   Labeled set
            features:   Features to be analyzed
            k:          Number of fold in cross-validation

        Return:
            pheno_df:   Labeled set
            X:          Labeled set input data
            y:          Labeled set labels
            X_train:    Training set input data
            X_test:     Test set input data
            y_train:    Training set labels
            y_test:     Test set labels
            phenotypes: List of phenotype classes
        """

    # Shuffle labeled set
    pheno_df = pheno_df.reindex(np.random.permutation(pheno_df.index)).reset_index(drop=True)
    pheno_df[features] = pheno_df[features].fillna(0)

    # Separate labeled set data and labels
    X = np.asarray(pheno_df[features])
    f = pd.factorize(pheno_df['phenotype'])
    y = np.zeros((f[0].shape[0], len(set(f[0]))))
    y[np.arange(f[0].shape[0]), f[0].T] = 1
    phenotypes = f[1]

    # Split training and test set for k-fold cross-validation
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    divide = X.shape[0] // k
    for cv in range(k):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (k-1):
            end = X.shape[0]
        mask_train = np.asarray([False if x in list(range(start, end)) else True for x in list(range(0, X.shape[0]))])
        X_train.append(X[mask_train == 1].copy())
        X_test.append(X[mask_train == 0].copy())
        y_train.append(y[mask_train == 1].copy())
        y_test.append(y[mask_train == 0].copy())

    return pheno_df, X, y, X_train, X_test, y_train, y_test, phenotypes


def make_predictions(df, param, pheno_df, threshold, features, output):
    """ Train neural network (NN) with cross-validation and make predictions.
        Save phenotype predictions and inlier/outlier info in df

        Args:
            df:         Dictionary of combined data
            param:      NN hyper-parameters
            pheno_df:   Labeled set
            threshold:  Probability threshold to make predictions
            features:   Features to be analyzed
            output:     Output filename

        Return:
            df:         Dictionary with phenotype predictions added
        """

    print('\nMaking predictions...')

    # Split training and test set for cross-validation
    k = param['k_fold_cv']
    pheno_df, X, y, X_train, X_test, y_train, y_test, phenotypes = split_labeled_set(pheno_df, features, k)

    # Initialize arrays for NN runs
    identifier_index = len(pheno_df.columns.values) - len(features)
    df_output = pheno_df.iloc[:, :identifier_index]
    sum_prob_labeled = np.zeros([y.shape[0], y.shape[1]])
    sum_prob_test = np.zeros([y.shape[0], y.shape[1]])

    # Train NN with cross validation for evaluating performance
    divide = X.shape[0] // k
    runn = 1
    for cv in range(k):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (k - 1):
            end = X.shape[0]
        # Train and make predictions for each fold for a number of runs
        for n in range(param['runs']):
            # Train NN with training set
            model = neural_network(X_train[cv], y_train[cv], param, phenotypes, X_test[cv], y_test[cv])
            # Predictions on test data
            probabilities_test = model.predict(X_test[cv], batch_size=param['batch_size'])
            sum_prob_test[start:end] += probabilities_test

            # Predictions on labeled data
            probabilities_labeled = model.predict(X, batch_size=param['batch_size'])
            predictions_labeled = np.argmax(probabilities_labeled, axis=1)
            sum_prob_labeled += probabilities_labeled
            df_output['Run-%d' % runn] = [phenotypes[i] for i in predictions_labeled]
            runn += 1

    # Train NN with the complete labeled set
    sum_prob_all = np.zeros([df['data_scaled'].shape[0], y.shape[1]])
    for n in range(param['runs']):
        model = neural_network(X, y, param, phenotypes)
        # Predictions on all data
        probabilities_all = model.predict(df['data_scaled'], batch_size=param['batch_size'])
        sum_prob_all += probabilities_all

    # Labeled set single cell accuracies
    cv_runs = k * param['runs']
    cell_accuracy(df_output, sum_prob_labeled, phenotypes, cv_runs, output)

    # Test-set predictions
    y_pred = np.argmax(sum_prob_test, axis=1)
    y_true = np.argmax(y, axis=1)
    plot_confusion_matrix(y_true, y_pred, phenotypes, output['Confusion'])

    # Make predictions for the complete data
    y_ALL = sum_prob_all/param['runs']
    y_prob_ALL = (y_ALL >= threshold).astype('int')
    y_pred_ALL = np.argmax(y_ALL, axis=1)
    phenotype_ALL = []
    for i in range(len(y_pred_ALL)):
        pred = phenotypes[y_pred_ALL[i]]
        # If none of the probabilities pass the threshold, predict as None phenotype
        if sum(y_prob_ALL[i]) == 0:
            pred = 'none'
        phenotype_ALL.append(pred)

    # Save phenotype predictions for cell_IDs provided
    cellID = pd.DataFrame(columns=['CellID', 'Prediction'] + list(phenotypes))
    cellID['CellID'] = df['cell_id']
    cellID['Prediction'] = np.array(phenotype_ALL)
    for i in range(len(phenotypes)):
        cellID[phenotypes[i]] = y_ALL[:, i]
    cellID = cellID.sort_values('CellID', ascending=True).reset_index(drop=True)
    cellID.to_csv(path_or_buf=output['PhenotypeCellIDs'], index=False)

    # Save predictions and inlier state in the combined dictionary
    df['phenotype'] = np.array(phenotype_ALL)
    df['is_inlier'] = np.array([p == 'negative' for p in df['phenotype']])

    return df


def neural_network(X_train, y_train, param, phenotypes, X_test=np.array([]), y_test=np.array([])):
    """ Train NN and return the model.

        Args:
            X_train:        Training set input data
            y_train:        Training set labels
            param:          Neural network hyper-parameters
            phenotypes:     List of phenotype classes
            X_test:         Test set input data
            y_test:         Test set labels

        Return:
            model:          Trained neural network
        """

    # NN layer units
    input_units = X_train.shape[1]
    output_units = len(phenotypes)

    # NN architecture
    model = Sequential()
    model.add(Dense(param['hidden_units'][0],
                    input_shape=(input_units,),
                    activation='relu'))
    model.add(Dense(param['hidden_units'][1],
                    activation='relu'))
    model.add(Dense(output_units,
                    activation='softmax'))
    sgd = optimizers.SGD(lr=param['learning_rate'],
                         decay=param['decay'],
                         momentum=param['momentum'],
                         nesterov=param['nesterov'])
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=param['num_epochs'],
              batch_size=param['batch_size'],
              validation_split=param['percent_to_valid'],
              verbose=1)

    # Evaluate model
    if len(X_test):
        score = model.evaluate(X_test, y_test, batch_size=param['batch_size'])
        print('Test %s: %.2f' % (model.metrics_names[0], score[0]))
        print('Test %s: %.2f%%\n' % (model.metrics_names[1], score[1] * 100))
    else:
        print('Trained on all labeled samples\n')

    return model


def cell_accuracy(df, sum_prob, phenotypes, n, output):
    """ Calculate accuracy for labeled set samples out of n runs.
        Include average probability for each phenotype.
        Save in an output file.

        Args:
            df:         Labeled set in a dataframe
            sum_prob:   Cumulative probability for each sample and label
            phenotypes: List of phenotype classes
            n:          Independent neural network training runs
            output:     Output filename
        """

    # Create columns for each cross-validation run
    df['accuracy'] = np.zeros(len(df))
    predictions = []
    for c in df.columns.values:
        if 'Run-' in c:
            predictions.append(df.columns.get_loc(c))

    # Calculate cell accuracy
    for i in range(len(df)):
        true_label = df.iloc[i, df.columns.get_loc('phenotype')]
        correct = 0
        for p in predictions:
            if true_label == df.iloc[i, p]:
                correct += 1
        df.iloc[i, df.columns.get_loc('accuracy')] = float(correct) / n

    # Calculate average probability for each phenotype class
    sum_prob = sum_prob / n
    for i in range(len(phenotypes)):
        df[phenotypes[i]] = sum_prob[:, i]

    # Save cell accuracy data
    df.to_csv(path_or_buf=output['CellAccuracy'], index=False)


def plot_confusion_matrix(y_true, y_pred, classes, output):
    """ Plot confusion matrix for all test set predictions.

        Args:
            y_true:     Actual labels
            y_pred:     Predicted labels
            classes:    List of phenotype labels
            output:     Output filename
        """

    # Normalize counts for each true-predicted label pair
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Acc %.2f%%' % (acc * 100))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    # Plot percentage of labeled samples in each true-predicted label pair
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    # Save plot
    fig = plt.gcf()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(output, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def pvalue_parameters(df):
    """ Return numbers of negative control cells and outlier negative control cells.
        These numbers are used to calculate hypergeometric p-value.

        Args:
            df:                 Dictionary of combined data

        Return:
            neg_cells:          Number of negative control cells
            neg_cells_outliers: Number of outlier negative control cells
        """

    neg_cells = len(df['mask_neg'][df['mask_neg'] == 1])
    neg_cells_outliers = len(df['mask_neg'][(df['mask_neg'] == 1) & (df['is_inlier'] == 0)])

    return neg_cells, neg_cells_outliers


def prepare_output_file_well(df, identifiers, phenotypes, output):
    """ Combine phenotype predictions and calculate penetrance for each well in a plate/arrayed format.

        Args:
            df:                 Dictionary of combined data
            identifiers:        Strain/condition identifier columns
            phenotypes:         List of phenotype classes
            output:             Output filename

        Return:
            final_df_output:    Results for each well
        """

    print('\nPreparing the output values...')

    # Save required data from dictionary to a pandas dataframe
    final_df = pd.DataFrame()
    for i in (identifiers + ['plate', 'row', 'column', 'phenotype', 'is_inlier']):
        final_df[i] = df[i]
    # Combine row and column information for a single well information
    final_df['well'] = final_df.row.map(int).map(str) + '_' + final_df.column.map(int).map(str)

    # Initialize output file columns
    output_columns = identifiers + ['plate', 'row', 'column', 'p_value', 'penetrance', 'num_cells']
    for i in phenotypes:
        output_columns.append(i)
    final_df_output = pd.DataFrame(columns=output_columns)
    this_row = 0

    # Extract negative control cell numbers for hypergeometric p-value calculation
    neg_cells, neg_cells_outliers = pvalue_parameters(df)

    # Analyze each plate separately
    for p in list(set(df['plate'])):
        final_df_plate = final_df[final_df['plate'] == p]

        # Analyze each well in each plate separately
        wells = final_df_plate['well'].unique().tolist()
        for well in wells:
            df_well = final_df_plate[final_df_plate['well'] == well]

            # Calculate penetrance (1 - negative%) and p-value
            is_inlier_well = np.asarray(df_well['is_inlier'])
            num_cells = df_well.shape[0]
            num_outliers = sum(is_inlier_well == 0)
            pene = float(num_outliers) / num_cells * 100
            pval = 1 - stats.hypergeom.cdf(num_outliers, neg_cells, neg_cells_outliers, num_cells)

            # Enter well results into a final dataframe row
            line = []
            for i in (identifiers + ['plate', 'row', 'column']):
                line.append(df_well[i].unique()[0])
            line.append(pval)
            line.append(pene)
            line.append(num_cells)
            for i in phenotypes:
                line.append(float(len(df_well[df_well.phenotype == i])) / num_cells)
            final_df_output.loc[this_row,] = line
            this_row += 1

    # Save results
    final_df_output = final_df_output.sort_values('plate', ascending=True).reset_index(drop=True)
    final_df_output.to_csv(path_or_buf=output['ODresultsWell'], index=False)

    return final_df_output


def prepare_output_file_strain(df, identifiers, identifier, phenotypes, output):
    """ Combine phenotype predictions and calculate penetrance for each unique strain/condition identifier.

        Args:
            df:                 Dictionary of combined data
            identifiers:        Strain/condition identifier columns
            identifier:         Unique identifier for the strain/condition
            phenotypes:         List of phenotype classes
            output:             Output filename

        Return:
            final_df_output:    Results for each strain/condition
        """

    # Save required data from dictionary to a pandas dataframe
    final_df = pd.DataFrame()
    for i in (identifiers + ['plate', 'row', 'column', 'phenotype', 'is_inlier']):
        final_df[i] = df[i]
    # Combine row and column information for a single well information
    final_df['well'] = final_df.plate + '_' + final_df.row.map(int).map(str) + '_' + final_df.column.map(int).map(str)

    # Initialize output file columns
    output_columns = identifiers + ['p_value', 'penetrance', 'num_cells', 'num_wells']
    for i in phenotypes:
        output_columns.append(i)
    final_df_output = pd.DataFrame(columns=output_columns)
    this_row = 0

    # Extract negative control cell numbers for hypergeometric p-value calculation
    neg_cells, neg_cells_outliers = pvalue_parameters(df)

    # Analyze each strain/condition separately
    strains = final_df[identifier].unique().tolist()
    for s in strains:
        df_strain = final_df[final_df[identifier] == s]
        is_inlier_strain = np.asarray(df_strain['is_inlier'])
        num_cells = df_strain.shape[0]

        # If there are no cells, skip results
        if num_cells == 0:
            print('Zero cells for %s' % s)

        # Calculate penetrance (1 - negative%) and p-value
        else:
            num_wells = len(df_strain['well'].unique())
            num_outliers = sum(is_inlier_strain == 0)
            pene = float(num_outliers) / num_cells * 100
            pval = 1 - stats.hypergeom.cdf(num_outliers, neg_cells, neg_cells_outliers, num_cells)

            # Enter strain/condition results into a final dataframe row
            line = []
            for i in identifiers:
                line.append(df_strain[i].unique()[0])
            line.append(pval)
            line.append(pene)
            line.append(num_cells)
            line.append(num_wells)
            for i in phenotypes:
                line.append(float(len(df_strain[df_strain.phenotype == i])) / num_cells)
            final_df_output.loc[this_row,] = line
            this_row += 1

    # Save results
    final_df_output = final_df_output.sort_values(identifier, ascending=True).reset_index(drop=True)
    final_df_output.to_csv(path_or_buf=output['ODresultsStrain'], index=False)

    return final_df_output


def evaluate_performance(controls_file, df, df_strain, neg, identifier, output):
    """ Extract positive controls to calculate TPR-FPR-Precision values.
        Predict penetrance bins if this information is provided in the controls file.

        Args:
            controls_file:  Controls file with positive and negative controls
            df:             Result dataframe for wells
            df_strain:      Result dataframe for strain/condition
            neg:            Negative controls for scaling
            identifier:     Unique identifier for the strain/condition
            output:         List of output filenames
        """

    print('\nEvaluating performances...')

    # Penetrance values of all positive control strains
    pos_controls = pd.read_csv(controls_file)
    pos_controls = lower_column_names(pos_controls)
    pos_control_strains = pos_controls[pos_controls['phenotype'] != 'negative'][identifier].unique().tolist()
    PC = []
    for strain in pos_control_strains:
        if strain in df_strain[identifier].tolist():
            PC.append(df_strain[df_strain[identifier] == strain]['penetrance'].values[0])
    PC = np.array(PC)

    # Penetrance values of all negative control wells
    NC = np.array([])
    for strain in neg:
        NC = np.append(NC, np.asarray(df[df[identifier] == strain]['penetrance'].values))

    # Evaluate TPR-FPR-Precision values for ROC and PR curves
    evaluate_performance_ROC_PR(NC, PC, output)

    # Predict penetrance bins if available
    evaluate_performance_penetrance_bins(controls_file, df_strain, identifier, output)


def evaluate_performance_ROC_PR(NC, PC, output):
    """ Calculate TPR-FPR-Precision values and save in an output file.
        Threshold changes from 0 to 100th percentile of negative control penetrance values.

        Args:
            NC:       Penetrance values of negative controls
            PC:       Penetrance values of positive controls
            output:   Output filename
        """

    # Calculate TPR-FPR-Precision values at each discrete negative control percentile
    performance = {'tpr': [], 'fpr': [], 'prec': []}
    penetrance_cutoff = []
    for i in range(101):
        # Penetrance at this percentile
        penetrance_threshold = stats.scoreatpercentile(NC, i)
        penetrance_cutoff.append(penetrance_threshold)
        # Count True and False Positives with each threshold
        TP = len(PC[PC >= penetrance_threshold])
        FP = len(NC[NC >= penetrance_threshold])
        # True positive rate (Recall) - TP / TP + FN
        performance['tpr'].append(TP / float(len(PC)))
        # False positive rate - FP / FP + TN
        performance['fpr'].append(FP / float(len(NC)))
        # Precision - TP / TP + FP
        if FP == 0:
            performance['prec'].append(1)
        else:
            performance['prec'].append(TP / float(TP + FP))

    # Save TPR-FPR-Precision values
    roc_pr = pd.DataFrame({'Neg_Percentile': np.asarray(range(101)),
                           'Penetrance_Cutoff': np.asarray(penetrance_cutoff),
                           'TPR (Recall)': np.asarray(performance['tpr']),
                           'FPR': np.asarray(performance['fpr']),
                           'Precision': np.asarray(performance['prec'])})
    roc_pr = roc_pr.sort_values('Neg_Percentile', ascending=False)
    roc_pr.to_csv(path_or_buf=output['ROCPRNumbers'], index=False)


def evaluate_performance_penetrance_bins(controls_file, df, identifier, output):
    """ Make predictions on penetrance bins provided in the controls file and save in an output file
        Label information:
        Bin-1: 80-100% penetrance
        Bin-2: 60-80% penetrance
        Bin-3: 40-60% penetrance
        Bin-4: 20-40% penetrance
        Bin-0: 0-20% penetrance

        Args:
            controls_file:  Controls file with controls and penetrance bin and manual penetrance labels
            df:             Result dataframe for strain/condition
            identifier:     Unique identifier for the strain/condition
            output:         Output filename
        """

    bin_df = pd.read_csv(controls_file)
    bin_df = lower_column_names(bin_df)

    if ('bin' in bin_df.columns.values):
        # Remove strains not screened
        mask = np.array([True if (x in df[identifier].tolist()) else False for x in bin_df[identifier]])
        bin_df = bin_df[mask == True]
        bin_df = bin_df.reset_index(drop=True)

        # Initialize output dataframe for penetrance bins
        bin_df_out = pd.DataFrame(columns=bin_df.columns.values.tolist() +
                                          ['penetrance', 'predicted_bin', 'p_value', 'num_cells', 'num_wells'])
        this_row = 0
        for i in range(len(bin_df)):
            # Extract strain/condition information and calculated penetrance (1 - negative%)
            strain = bin_df.iloc[i, bin_df.columns.get_loc(identifier)]
            penetrance = df[df[identifier] == strain]['penetrance'].values[0]

            # Predict penetrance bin
            if (penetrance > 80) and (penetrance <= 100):
                predicted_bin = 1
            elif (penetrance > 60) and (penetrance <= 80):
                predicted_bin = 2
            elif (penetrance > 40) and (penetrance <= 60):
                predicted_bin = 3
            elif (penetrance > 20) and (penetrance <= 40):
                predicted_bin = 4
            else:
                predicted_bin = 0

            # Gather ppredicted penetrance bin and other information
            line = bin_df.iloc[i, :].tolist()
            line.append(penetrance)
            line.append(predicted_bin)
            line.append(df[df[identifier] == strain]['p_value'].values[0])
            line.append(df[df[identifier] == strain]['num_cells'].values[0])
            line.append(df[df[identifier] == strain]['num_wells'].values[0])
            bin_df_out.loc[this_row,] = line
            this_row += 1

        # Save results
        bin_df_out.to_csv(path_or_buf=output['PenetranceBins'], index=False)

    else:
        print('\nNo penetrance bins for this screen!')

    print('\n\n')

