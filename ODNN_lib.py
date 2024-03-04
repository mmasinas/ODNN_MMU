import os
import sys
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers, losses
from keras.layers import Dense
from keras.models import Sequential
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg')


def read_input_files(input_files, input_data_file, features_file, mapping_file, location, strain_identifiers):
    """ Read input files for plates, CP features, mapping sheet.
        Return in lists and dataframe.

        Args:
            input_files (str): File of plate list to be analyzed
            input_data_file (str): Input data, scaled and filled with identifiers
            features_file (str): File of CP feature list to be used in the analysis
            mapping_file (str): Map gene names from plate-row-column info.
            location (list): Column names for plate-row-column info.

        Returns:
            plates (list): Plates to be analyzed
            feature_set (list): Features to be analyzed
            mapping_sheet (pd.DataFrame): Dataframe to map strain/condition information from arrayed plates
            map_features (list): Strain/condition identifier columns
        """

    # Read plate list
    plates = []
    if os.path.isfile(input_files):
        f = open(input_files, 'r')
        plates = list(filter(None, [x.strip() for x in f.readlines()]))
        f.close()
    elif not os.path.isfile(input_data_file):
        sys.exit('Please provide a list of plate paths to analyze, quitting...')

    # Read CP feature list
    if os.path.isfile(features_file):
        f = open(features_file, 'r')
        feature_set = list(filter(None, [x.strip() for x in f.readlines()]))
        feature_set = [f.lower() for f in feature_set]
        f.close()
    else:
        sys.exit('Please provide a CP feature list, quitting...')

    # Read strain/condition information file
    if os.path.isfile(mapping_file):
        mapping_sheet = lower_column_names(pd.read_csv(mapping_file)).fillna('')
        map_features = mapping_sheet.columns.values.tolist()
        for i in location:
            map_features.remove(i)
    else:
        if (os.path.isfile(input_data_file)) or (os.path.isfile(input_files) and strain_identifiers):
            mapping_sheet = None
            map_features = [s.lower() for s in strain_identifiers.split(',')]
            print('No mapping sheet is found. The process will continue assuming the input is already scaled.')
        else:
            sys.exit('Please provide a mapping sheet for strain/condition information, quitting...')

    return plates, feature_set, mapping_sheet, map_features


def lower_column_names(df):
    """ Return pandas dataframe with lower column names.

        Args:
            df (pd.DataFrame): Pandas Dataframe object

        Returns:
            df (pd.DataFrame): Pandas Dataframe object with lowercase column names
        """

    df.columns = [c.lower() for c in df.columns]

    return df


def read_negative_controls_file(filename, identifier):
    """ Read controls file and return negative controls.

        Args:
            filename (str): Controls file with positive and negative controls
            identifier (str): Unique identifier for the strain/condition

        Returns:
            neg_controls (list): List of negative controls
        """

    if os.path.isfile(filename):
        neg_df = pd.read_csv(filename)
        neg_df = lower_column_names(neg_df)
        neg_df = neg_df[neg_df['phenotype'] == 'negative']
        neg_controls = neg_df[identifier].values

    else:
        sys.exit('Please provide a controls file, quitting...')

    return neg_controls


def initialize_dictionary(identifiers, location_feat):
    """ Return an empty dictionary with the following attributes.

        Args:
            identifiers (list): Strain/condition identifier columns
            location_feat (list): Column names for plate-row-column info

        Returns:
            main_dict (dict): Dictionary of combined data
            dict_feat (dict): Dictionary features
        """

    main_dict = {}
    for i in (identifiers + ['cell_id'] + location_feat + ['data', 'data_scaled', 'mask_neg']):
        main_dict[i] = np.array([])

    dict_feat = list(main_dict.keys())

    return main_dict, dict_feat


def read_and_scale_plate(main_dict, plate, neg, features, mapping_sheet, identifier, identifiers, dict_feat):
    """ Read, scale and add plate to the existing combined dictionary.

        Args:
            main_dict (dict): Dictionary of combined data
            plate (str): Plate filename
            neg (list): Negative controls for scaling
            features (list): Features to be analyzed
            mapping_sheet (pd.DataFrame): Strain information on plates
            identifier (str): Unique identifier for the strain/condition
            identifiers (list): Strain/condition identifier columns
            dict_feat(list): Dictionary features

        Returns:
            main_dict (dict): Dictionary with new plate data appended
        """

    print('Reading: %s' % plate)

    # Read plate
    df = pd.read_csv(plate, header=0)
    df = lower_column_names(df)
    df['plate'] = plate
    dict_plate = extract_plate_information(df, plate, features, mapping_sheet, neg, identifier, identifiers)

    # Scale plate wrt to negative control cells
    data_wt = dict_plate['data'][dict_plate['mask_neg'] == 1]
    dict_plate['data_scaled'] = standard_scaler_fit_transform(data_wt, dict_plate['data'])
    dict_plate['data_scaled'][np.isnan(dict_plate['data_scaled'])] = 0

    # Add to the existing dictionary of combined data
    main_dict = append_data(main_dict, dict_plate, dict_feat, 'data')

    return main_dict


def extract_plate_information(df, filename, features, mapping_sheet, neg, identifier, identifiers):
    """ Read a CP output for a plate and extract required information.

        Args:
            df (pd.DataFrame): CP output of a plate as a dataframe
            filename (str): Filename of the plate
            features (list): Features to be analyzed
            mapping_sheet(pd.DataFrame): Dataframe to map gene and allele names
            neg (list): Negative controls for scaling
            identifier (str): Unique identifier for the strain/condition
            identifiers (list): Strain/condition identifier columns

        Returns:
            dict_plate (dict): Extracted plate information as a dictionary
        """

    # Extract plate locations and cell_IDs from CP output
    dict_plate, _ = initialize_dictionary(identifiers)
    dict_plate['cell_id'] = np.array(df['cell_id'], dtype='int32')
    dict_plate['column'] = np.array(df['col'], dtype='int32')
    dict_plate['row'] = np.array(df['row'], dtype='int32')
    dict_plate['plate'] = np.array(df['plate'])

    # Map strain/condition information from arrayed format
    plate_name = filename.split('/')[-1].split('_')[0].replace('plate', '')
    ms_plate = mapping_sheet[mapping_sheet.plate == plate_name]

    # Initialize strain identifier lists
    identifier_lists = [[] for _ in range(len(identifiers))]

    # Add strain identifier information for each well
    for i in range(len(dict_plate['column'])):
        map_df = ms_plate[(ms_plate['column'] == dict_plate['column'][i]) & (ms_plate['row'] == dict_plate['row'][i])]
        for j in range(len(identifier_lists)):
            identifier_lists[j].append(map_df[identifiers[j]].values[0])

    # Add strain identifier information to the plate dictionary data
    for i in range(len(identifiers)):
        dict_plate[identifiers[i]] = np.array(identifier_lists[i])

    # Keep CP features needed for analysis
    dict_plate['data'] = np.array(df[features], dtype='float64')

    # Find the negative control cells in the data
    dict_plate['mask_neg'] = np.array([x in neg for x in dict_plate[identifier]])

    return dict_plate


def standard_scaler_fit_transform(data_fit, data_transform=np.array([])):
    """ Scale data into zero mean and unit variance for each column.

        Args:
            data_fit (np.array): data to calculate mean and covariances
            data_transform (np.array): data to scale wrt data_fit

        Returns:
            Scaled data (np.array)
        """

    mean = np.nanmean(data_fit, axis=0)
    variance = np.nanstd(data_fit, axis=0)

    if len(data_transform) > 0:
        return (data_transform - mean) / variance
    else:
        return (data_fit - mean) / variance


def save_data(main_dict, features, location_feat, identifiers, output):
    """ Save combined scale data for all plates.

        Args:
            main_dict (dict): Dictionary of combined data
            features (list): Features to be analyzed
            location_feat (list): Column names for plate-row-column info
            identifiers (list): Strain/condition identifier columns
            output (dict): Output filenames
        """

    # Save scaled data for all plates
    data_scaled_columns = ['cell_id'] + location_feat + identifiers + features
    data_scaled_output = main_dict['cell_id'].reshape(-1, 1)
    for i in (location_feat + identifiers):
        data_scaled_output = np.concatenate((data_scaled_output, main_dict[i].reshape(-1, 1)), axis=1)
    data_scaled_output = np.concatenate((data_scaled_output, main_dict['data_scaled']), axis=1)
    data_scaled_output_df = pd.DataFrame(data=data_scaled_output, columns=data_scaled_columns)
    data_scaled_output_df.to_csv(path_or_buf=output['DataScaled'], index=False)


def append_data(main_dict, dict_add, dict_feat, data_type):
    """ Append two dictionaries on selected features.

        Args:
            main_dict (dict): Dictionary of combined data
            dict_add (dict): Dictionary to be added
            dict_feat (list): Dictionary features

        Returns:
            main_dict (dict): Dictionary of combined data
        """

    # Create a deep copy of features
    append_list = dict_feat[:]
    if len(main_dict[data_type]) == 0:
        main_dict['data'] = dict_add['data']
        main_dict['data_scaled'] = dict_add['data_scaled']
        if data_type == 'data_scaled':
            main_dict['mask_neg'] = dict_add['mask_neg']
        append_list.remove('data')
        append_list.remove('data_scaled')
        append_list.remove('mask_neg')

    # Append new dictionary to the existing one
    for i in append_list:
        main_dict[i] = np.append(main_dict[i], dict_add[i], axis=0)

    return main_dict


def read_scaled_data(p, main_dict, input_data, dict_feat, neg, identifier, features,
                     identifiers, location_feat, multi_plate=False):
    """ Read an input file with scaled CP features and strain/condition identifier information.

        Args:
            main_dict (dict): Initialized dictionary
            input_data (str): Input data, scaled and filled with identifiers
            dict_feat (list): Dictionary features
            neg (list): Negative controls for scaling
            identifier (str): Unique identifier for the strain/condition
            features (list): Features to be analyzed
            identifiers (list): Strain/condition identifier columns
            location_feat (list): Column names for plate-row-column info
            multi_plate (bool): Whether processing multiple input plates or not

        Returns:
            main_dict (dict): Dictionary of combined data
        """

    if multi_plate:
        print('Returning scaled data from %s' % p)
    else:
        print('Returning scaled data from %s' % input_data)

    # Read scaled data and save in a dataframe
    df_scaled = pd.read_csv(p)
    df_scaled = lower_column_names(df_scaled)
    exclude = ['data', 'data_scaled', 'mask_neg']

    if multi_plate:
        dict_plate, dict_feat = initialize_dictionary(identifiers, location_feat)
    else:
        dict_plate = main_dict

    for i in dict_feat:
        if i not in exclude:
            dict_plate[i] = np.array(df_scaled[i])

    # Use the original order of features
    dict_plate['data_scaled'] = np.array(df_scaled[features])
    dict_plate['data_scaled'][np.isnan(dict_plate['data_scaled'])] = 0
    dict_plate['mask_neg'] = np.array([x in neg for x in dict_plate[identifier]])

    if multi_plate:
        # Add to the existing dictionary of combined data
        main_dict = append_data(main_dict, dict_plate, dict_feat, 'data_scaled')

    return main_dict


def prepare_phenotype_data(main_dict, identifier, identifiers, features, location_feat, pos_controls_files, output):
    """ Read and prepare training set from phenotype data with positive controls.

        Args:
            main_dict (dict): Dictionary of combined data
            identifier (str): Unique identifier for the strain/condition
            identifiers (list): Strain/condition identifier columns
            features (list): Features to be analyzed
            location_feat (list): Column names for plate-row-column info
            pos_controls_files (list): List of positive controls file
            output (dict): Output filenames

        Returns:
            main_dict (dict): Dictionary with positive control information added
            phenotype_df (pd.DataFrame): Labeled set as a dataframe
            phenotype_classes (list): List of phenotype classes
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
            main_dict['mask_pc'] = np.array([x in pc['cell_id'].values for x in main_dict['cell_id']])
            phenotypes = [pc[pc['cell_id'] == x]['phenotype'].values[0]
                          for x in main_dict['cell_id'][main_dict['mask_pc'] == 1]]

        # If training set with strain/condition is available, use all cells
        elif pos_controls_strain:
            print('\nPreparing phenotype data with strain labeled set...')
            pc = pd.read_csv(pos_controls_strain)
            pc = lower_column_names(pc)
            phenotype_classes = pc['phenotype'].unique()
            main_dict['mask_pc'] = np.array([x in pc[identifier].values for x in main_dict[identifier]])
            phenotypes = [pc[pc[identifier] == x]['phenotype'].values[0]
                          for x in main_dict[identifier][main_dict['mask_pc'] == 1]]

        # Combine positive control data
        phenotype_data_columns = ['cell_id'] + identifiers + location_feat + ['phenotype'] + features
        phenotype_data = main_dict['cell_id'][main_dict['mask_pc'] == 1].reshape(-1, 1)
        for i in (identifiers + location_feat):
            phenotype_data = np.concatenate((phenotype_data, main_dict[i][main_dict['mask_pc'] == 1].reshape(-1, 1)),
                                            axis=1)
        phenotype_data = np.concatenate((phenotype_data, np.array(phenotypes).reshape(-1, 1),
                                         main_dict['data_scaled'][main_dict['mask_pc'] == 1]), axis=1)
        phenotype_df = pd.DataFrame(data=phenotype_data, columns=phenotype_data_columns)

    # Save positive control data
    phenotype_df = phenotype_df.fillna('').reset_index(drop=True)
    phenotype_df.to_csv(path_or_buf=output['PhenotypeData'], index=False)

    # Add None as a phenotype class
    phenotype_classes = np.append(phenotype_classes, ['none'])

    return main_dict, phenotype_df, phenotype_classes


def split_labeled_set(pheno_df, features, k):
    """ Split labeled set into training and test set for k-fold cross-validation

        Args:
            pheno_df (pd.DataFrame): Labeled set
            features (list): Features to be analyzed
            k (int): Number of fold in cross-validation

        Returns:
            pheno_df (pd.DataFrame): Labeled set
            X (np.array): Labeled set input data
            y (np.array): Labeled set labels
            X_train (np.array): Training set input data
            X_test (np.array): Test set input data
            y_train (np.array): Training set labels
            y_test (np.array): Test set labels
            phenotypes (list): List of phenotype classes
        """

    # Shuffle labeled set
    pheno_df = pheno_df.reindex(np.random.permutation(pheno_df.index)).reset_index(drop=True)
    pheno_df[features] = pheno_df[features].fillna(0)

    # Separate labeled set data and labels
    x = np.asarray(pheno_df[features])
    f = pd.factorize(pheno_df['phenotype'])
    y = np.zeros((f[0].shape[0], len(set(f[0]))))
    y[np.arange(f[0].shape[0]), f[0].T] = 1
    phenotypes = f[1]

    # Split training and test set for k-fold cross-validation
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    divide = x.shape[0] // k
    for cv in range(k):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (k - 1):
            end = x.shape[0]
        mask_train = np.asarray([False if x in list(range(start, end)) else True for x in list(range(0, x.shape[0]))])
        x_train.append(x[mask_train == 1].copy())
        x_test.append(x[mask_train == 0].copy())
        y_train.append(y[mask_train == 1].copy())
        y_test.append(y[mask_train == 0].copy())

    return pheno_df, x, y, x_train, x_test, y_train, y_test, phenotypes


def make_predictions(main_dict, param, pheno_df, threshold, features, output):
    """ Train neural network (NN) with cross-validation and make predictions.
        Save phenotype predictions and inlier/outlier info in df

        Args:
            main_dict (dict): Dictionary of combined data
            param (dict): NN hyper-parameters
            pheno_df (pd.DataFrame): Labeled set
            threshold (float): Probability threshold to make predictions
            features (list): Features to be analyzed
            output (dict): Output filenames

        Returns:
            main_dict (dict): Dictionary with phenotype predictions added
        """

    print('\nMaking predictions...')

    # Split training and test set for cross-validation
    pheno_df, x, y, x_train, x_test, y_train, y_test, phenotypes = split_labeled_set(pheno_df, features,
                                                                                     param['k_fold_cv'])

    # Initialize arrays for NN runs
    identifier_index = len(pheno_df.columns.values) - len(features)
    df_output = pheno_df.iloc[:, :identifier_index]
    sum_prob_labeled = np.zeros([y.shape[0], y.shape[1]])
    sum_prob_test = np.zeros([y.shape[0], y.shape[1]])

    # Train NN with cross validation for evaluating performance
    performance = pd.DataFrame()
    divide = x.shape[0] // param['k_fold_cv']
    run = 1
    for cv in range(param['k_fold_cv']):
        start = cv * divide
        end = (cv + 1) * divide
        if cv == (param['k_fold_cv'] - 1):
            end = x.shape[0]
        # Train and make predictions for each fold for a number of runs
        for n in range(param['runs']):
            runn = n + cv * param['runs']
            # Train NN with training set
            print("Training on training set, run #%d of %d" % (n, param['runs']))
            model, performance = neural_network(x_train[cv], y_train[cv], param, phenotypes, performance, runn,
                                                x_test[cv], y_test[cv])
            # Predictions on test data
            probabilities_test = model.predict(x_test[cv], batch_size=param['batch_size'])
            sum_prob_test[start:end] += probabilities_test

            # Predictions on labeled data
            probabilities_labeled = model.predict(x, batch_size=param['batch_size'])
            predictions_labeled = np.argmax(probabilities_labeled, axis=1)
            sum_prob_labeled += probabilities_labeled
            df_output['Run-%d' % run] = [phenotypes[i] for i in predictions_labeled]
            run += 1

    # Save training performance of cross-validation
    num_runs = param['k_fold_cv'] * param['runs']
    plot_training_performance(performance, output['TrainingCV'], num_runs)

    # Train NN with the complete labeled set
    performance = pd.DataFrame()
    sum_prob_all = np.zeros([main_dict['data_scaled'].shape[0], y.shape[1]])
    for n in range(param['runs']):
        print("Training on labeled set, run #%d of %d" % (n, param['runs']))
        model, performance = neural_network(x, y, param, phenotypes, performance, n)
        # Predictions on all data
        probabilities_all = model.predict(main_dict['data_scaled'], batch_size=param['batch_size'])
        sum_prob_all += probabilities_all
    plot_training_performance(performance, output['Training'], param['runs'])

    # Labeled set single cell accuracies
    cell_accuracy(df_output, sum_prob_labeled, phenotypes, num_runs, output)

    # Test-set predictions
    y_pred = np.argmax(sum_prob_test, axis=1)
    y_true = np.argmax(y, axis=1)
    plot_confusion_matrix(y_true, y_pred, phenotypes, output['Confusion'])

    # Make predictions for the complete data
    y_all = sum_prob_all / param['runs']
    y_prob_all = (y_all >= threshold).astype('int')
    y_pred_all = np.argmax(y_all, axis=1)
    phenotype_all = []
    for i in range(len(y_pred_all)):
        pred = phenotypes[y_pred_all[i]]
        # If none of the probabilities pass the threshold, predict as None phenotype
        if sum(y_prob_all[i]) == 0:
            pred = 'none'
        phenotype_all.append(pred)
    
    # Save phenotype predictions for cell_IDs provided
    cell_id = pd.DataFrame(columns=['CellID', 'Prediction'] + list(phenotypes))
    cell_id['CellID'] = main_dict['cell_id']
    cell_id['Prediction'] = np.array(phenotype_all)
    for i in range(len(phenotypes)):
        cell_id[phenotypes[i]] = y_all[:, i]
    cell_id = cell_id.sort_values('CellID', ascending=True).reset_index(drop=True)
    cell_id.to_csv(path_or_buf=output['PhenotypeCellIDs'], index=False)

    # Save predictions and inlier state in the combined dictionary
    main_dict['phenotype'] = np.array(phenotype_all)
    main_dict['is_inlier'] = np.array([p == 'negative' for p in main_dict['phenotype']])

    return main_dict


def neural_network(x_train, y_train, param, phenotypes, performance, n, x_test=np.array([]), y_test=np.array([])):
    """ Train NN and return the model.

        Args:
            x_train (np.array): Training set input data
            y_train (np.array): Training set labels
            param (dict): Neural network hyper-parameters
            phenotypes (list): List of phenotype classes
            performance (pd.DataFrame): Cross-entropy and accuracy at each training
            n (int): The specific run out of the total random initializations
            x_test (np.array): Test set input data
            y_test (np.array): Test set labels

        Returns:
            model (keras.models.Sequential): Trained neural network
        """

    # NN layer units
    input_units = x_train.shape[1]
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
    hist = model.fit(x_train, y_train,
                     epochs=param['num_epochs'],
                     batch_size=param['batch_size'],
                     validation_split=param['percent_to_valid'],
                     verbose=1)
    
    # Evaluate model
    performance['Loss_%d' % n] = hist.history['loss']
    performance['Val_Loss_%d' % n] = hist.history['val_loss']
    performance['Accuracy_%d' % n] = hist.history['accuracy']
    performance['Val_Accuracy_%d' % n] = hist.history['val_accuracy']

    if len(x_test):
        score = model.evaluate(x_test, y_test, batch_size=param['batch_size'])
        print('Test %s: %.2f' % (model.metrics_names[0], score[0]))
        print('Test %s: %.2f%%\n' % (model.metrics_names[1], score[1] * 100))
    else:
        print('Trained on all labeled samples\n')

    return model, performance


def plot_training_performance(performance, output, num_runs):
    """ Plot the cross-entropy and accuracy for training and validation set.

        Args:
            performance (pd.DataFrame): Cross-entropy and accuracy at each training
            output (dict): Output filenames
            num_runs (int): Total number of runs (number of random initializations times the number of folds)
        """

    # Save the training performance in a spreadsheet
    performance.to_csv('%s.csv' % output, index=False)

    fontsize = 16
    plt.figure(figsize=(10, 10))

    # Loss
    plt.subplot(211)
    train_all = []
    valid_all = []

    for i in range(num_runs):
        train = performance.iloc[:, performance.columns.get_loc('Loss_%d' % i)].values
        valid = performance.iloc[:, performance.columns.get_loc('Val_Loss_%d' % i)].values
        train_all.append(train)
        valid_all.append(valid)
        plt.plot(train, 'lightblue', alpha=0.4)
        plt.plot(valid, 'lightgreen', alpha=0.4)

    plt.plot(np.mean(train_all, axis=0), 'blue', label='Training')
    plt.plot(np.mean(valid_all, axis=0), 'green', label='Validation')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(fontsize=fontsize-4)
    fig = plt.gcf()
    plt.legend(fontsize=fontsize, loc='upper right')

    # Acc
    plt.subplot(212)
    train_all = []
    valid_all = []
    for i in range(num_runs):
        train = performance.iloc[:, performance.columns.get_loc('Accuracy_%d' % i)].values
        valid = performance.iloc[:, performance.columns.get_loc('Val_Accuracy_%d' % i)].values
        train_all.append(train)
        valid_all.append(valid)
        plt.plot(train, 'lightblue', alpha=0.4)
        plt.plot(valid, 'lightgreen', alpha=0.4)

    plt.plot(np.mean(train_all, axis=0), 'blue', label='Training')
    plt.plot(np.mean(valid_all, axis=0), 'green', label='Validation')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(fontsize=fontsize-4)
    plt.ylim([0, 1.1])
    plt.legend(fontsize=fontsize, loc='lower right')

    fig.savefig('%s.png' % output, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def cell_accuracy(df, sum_prob, phenotypes, n, output):
    """ Calculate accuracy for labeled set samples out of n runs.
        Include average probability for each phenotype.
        Save in an output file.

        Args:
            df (pd.DataFrame): Labeled set in a dataframe
            sum_prob (np.array): Cumulative probability for each sample and label
            phenotypes (list): List of phenotype classes
            n (int): Independent neural network training runs
            output (dict): Output filenames
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
            y_true (np.array): Actual labels
            y_pred (np.array): Predicted labels
            classes (list): List of phenotype labels
            output (dict): Output filenames
        """

    # Sort class labels
    class_list = classes.tolist()
    sorted_labels = sorted(class_list)
    sorted_labels_idx = [class_list.index(c) for c in sorted_labels]

    # Normalize counts for each true-predicted label pair
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=sorted_labels_idx)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Acc %.2f%%' % (acc * 100))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, sorted_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, sorted_labels)

    # Plot percentage of labeled samples in each true-predicted label pair
    thresh = np.max(cm) / 2.
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


def pvalue_parameters(main_dict):
    """ Return numbers of negative control cells and outlier negative control cells.
        These numbers are used to calculate hyper-geometric p-value.

        Args:
            main_dict (dict): Dictionary of combined data

        Returns:
            neg_cells (int): Number of negative control cells
            neg_cells_outliers (int): Number of outlier negative control cells
        """

    neg_cells = len(main_dict['mask_neg'][main_dict['mask_neg'] == 1])
    neg_cells_outliers = len(main_dict['mask_neg'][(main_dict['mask_neg'] == 1) & (main_dict['is_inlier'] == 0)])

    return neg_cells, neg_cells_outliers


def prepare_output_file_well(main_dict, identifiers, location_feat, phenotypes, output):
    """ Combine phenotype predictions and calculate penetrance for each well in a plate/arrayed format.

        Args:
            main_dict (dict): Dictionary of combined data
            identifiers (list): Strain/condition identifier columns
            location_feat (list): Column names for plate-row-column info
            phenotypes (list): List of phenotype classes
            output (dict): Output filenames

        Returns:
            df_output (pd.DataFrame): Results for each well
        """

    print('\nPreparing the output values...')

    # Save required data from dictionary to a pandas dataframe
    df = pd.DataFrame()
    for i in (identifiers + location_feat + ['phenotype', 'is_inlier']):
        df[i] = main_dict[i]
    # Combine row and column information for a single well information
    row_str = location_feat[1]
    col_str = location_feat[2]
    df['well'] = df[row_str].map(int).map(str) + '_' + df[col_str].map(int).map(str)

    # Initialize output file columns
    output_columns = identifiers + location_feat + ['p_value', 'penetrance', 'num_cells']
    for i in phenotypes:
        output_columns.append(i)
    df_output = pd.DataFrame(columns=output_columns)
    this_row = 0

    # Extract negative control cell numbers for hyper-geometric p-value calculation
    neg_cells, neg_cells_outliers = pvalue_parameters(main_dict)

    # Analyze each plate separately
    for plate in list(set(main_dict['plate'])):
        df_plate = df[df['plate'] == plate]

        # Analyze each well in each plate separately
        for well in df_plate['well'].unique():
            df_well = df_plate[df_plate['well'] == well]

            # Calculate penetrance (1 - negative%) and p-value
            is_inlier_well = np.asarray(df_well['is_inlier'])
            num_cells = df_well.shape[0]
            num_outliers = sum(is_inlier_well == 0)
            pene = float(num_outliers) / num_cells * 100
            pval = 1 - stats.hypergeom.cdf(num_outliers, neg_cells, neg_cells_outliers, num_cells)

            # Enter well results into a final dataframe row
            line = []
            for i in (identifiers + location_feat):
                line.append(df_well[i].unique()[0])
            line.append(pval)
            line.append(pene)
            line.append(num_cells)
            for i in phenotypes:
                line.append(float(len(df_well[df_well.phenotype == i])) / num_cells)
            df_output.loc[this_row] = line
            this_row += 1

    # Save results
    df_output = df_output.sort_values('plate', ascending=True).reset_index(drop=True)
    df_output.to_csv(path_or_buf=output['ODresultsWell'], index=False)

    return df_output


def prepare_output_file_strain(main_dict, identifiers, identifier, location_feat, phenotypes, output):
    """ Combine phenotype predictions and calculate penetrance for each unique strain/condition identifier.

        Args:
            main_dict (dict): Dictionary of combined data
            identifiers (list): Strain/condition identifier columns
            identifier (str): Unique identifier for the strain/condition
            location_feat (list): Column names for plate-row-column info
            phenotypes (list): List of phenotype classes
            output (dict): Output filenames

        Returns:
            df_output (pd.DataFrame): Results for each strain/condition
        """

    # Save required data from dictionary to a pandas dataframe
    df = pd.DataFrame()
    for i in (identifiers + location_feat + ['phenotype', 'is_inlier']):
        df[i] = main_dict[i]
    # Combine row and column information for a single well information
    #df['well'] = df.plate + '_' + df.row.map(int).map(str) + '_' + df.column.map(int).map(str)
    plate_str = location_feat[0]
    row_str = location_feat[1]
    col_str = location_feat[2]
    df['well'] = df[plate_str].astype(str) + '_' + df[row_str].astype(str) + '_' + df[col_str].astype(str)

    # Initialize output file columns
    output_columns = identifiers + ['p_value', 'penetrance', 'num_cells', 'num_wells']
    for i in phenotypes:
        output_columns.append(i)
    df_output = pd.DataFrame(columns=output_columns)
    this_row = 0

    # Extract negative control cell numbers for hyper-geometric p-value calculation
    neg_cells, neg_cells_outliers = pvalue_parameters(main_dict)

    # Analyze each strain/condition separately
    for strain in df[identifier].unique():
        df_strain = df[df[identifier] == strain]
        is_inlier_strain = np.asarray(df_strain['is_inlier'])
        num_cells = df_strain.shape[0]

        # If there are no cells, skip results
        if num_cells == 0:
            print('Zero cells for %s' % strain)

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
            df_output.loc[this_row] = line
            this_row += 1

    # Save results
    df_output = df_output.sort_values(identifier, ascending=True).reset_index(drop=True)
    df_output.to_csv(path_or_buf=output['ODresultsStrain'], index=False)

    return df_output


def evaluate_performance(controls_file, df_well, df_strain, neg, identifier, output):
    """ Extract positive controls to calculate TPR-FPR-Precision values.
        Predict penetrance bins if this information is provided in the controls file.

        Args:
            controls_file (str): Controls file with positive and negative controls
            df_well (pd.DataFrame): Result dataframe for wells
            df_strain (pd.DataFrame): Result dataframe for strain/condition
            neg (list): Negative controls for scaling
            identifier (str): Unique identifier for the strain/condition
            output (dict):  List of output file names
        """

    print('\nEvaluating performances...')

    # Penetrance values of all positive control strains
    pos_controls = pd.read_csv(controls_file)
    pos_controls = lower_column_names(pos_controls)
    pc = []
    for strain in pos_controls[pos_controls['phenotype'] != 'negative'][identifier].unique():
        if strain in df_strain[identifier]:
            pc.append(df_strain[df_strain[identifier] == strain]['penetrance'].values[0])
    pc = np.array(pc)

    # Penetrance values of all negative control wells
    nc = np.array([])
    for strain in neg:
        nc = np.append(nc, np.asarray(df_well[df_well[identifier] == strain]['penetrance'].values))

    # Evaluate TPR-FPR-Precision values for ROC and PR curves
    evaluate_performance_roc_pr(nc, pc, output)

    # Predict penetrance bins if available
    evaluate_performance_penetrance_bins(controls_file, df_strain, identifier, output)


def evaluate_performance_roc_pr(nc, pc, output):
    """ Calculate TPR-FPR-Precision values and save in an output file.
        Threshold changes from 0 to 100th percentile of negative control penetrance values.

        Args:
            nc (np.array): Penetrance values of negative controls
            pc(np.array): Penetrance values of positive controls
            output (dict): Output filenames
        """

    # Calculate TPR-FPR-Precision values at each discrete negative control percentile
    performance = {'tpr': [], 'fpr': [], 'prec': []}
    penetrance_cutoff = []
    for i in range(101):
        # Penetrance at this percentile
        penetrance_threshold = stats.scoreatpercentile(nc, i)
        penetrance_cutoff.append(penetrance_threshold)
        # Count True and False Positives with each threshold
        tp = len(pc[pc >= penetrance_threshold])
        fp = len(nc[nc >= penetrance_threshold])
        # True positive rate (Recall) - TP / TP + FN
        performance['tpr'].append(tp / float(len(pc)))
        # False positive rate - FP / FP + TN
        performance['fpr'].append(fp / float(len(nc)))
        # Precision - TP / TP + FP
        if fp:
            performance['prec'].append(tp / float(tp + fp))

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
            controls_file (str): Controls file with controls and penetrance bin and manual penetrance labels
            df (pd.DataFrame): Result dataframe for strain/condition
            identifier (str): Unique identifier for the strain/condition
            output (dict): Output filenames
        """

    bin_df = pd.read_csv(controls_file)
    bin_df = lower_column_names(bin_df)

    if 'bin' in bin_df.columns.values:
        # Remove strains not screened
        mask = np.array([1 if (x in df[identifier]) else False for x in bin_df[identifier]])
        bin_df = bin_df[mask == 1]
        bin_df = bin_df.reset_index(drop=True)

        # Initialize output dataframe for penetrance bins
        bin_df_out = pd.DataFrame(columns=bin_df.columns.values.tolist() + ['penetrance', 'predicted_bin', 'p_value',
                                                                            'num_cells', 'num_wells'])
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

            # Gather predicted penetrance bin and other information
            line = bin_df.iloc[i, :].tolist()
            line.append(penetrance)
            line.append(predicted_bin)
            line.append(df[df[identifier] == strain]['p_value'].values[0])
            line.append(df[df[identifier] == strain]['num_cells'].values[0])
            line.append(df[df[identifier] == strain]['num_wells'].values[0])
            bin_df_out.loc[this_row, ] = line
            this_row += 1

        # Save results
        bin_df_out.to_csv(path_or_buf=output['PenetranceBins'], index=False)

    else:
        print('\nNo penetrance bins for this screen!')

    print('\n\n')
