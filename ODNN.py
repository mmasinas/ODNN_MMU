import argparse
from ODNN_lib import *

if __name__ == '__main__':
    # Parameters to specify
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-files', default='',
                        help='Input files to be analyzed')
    parser.add_argument('-d', '--input-data', default='',
                        help='Input data, scaled and filled with identifiers')
    parser.add_argument('-f', '--features-file', default='',
                        help='Feature sets to include')
    parser.add_argument('-m', '--mapping-file', default='',
                        help='Mapping sheet for strain identifiers')
    parser.add_argument('-p', '--probability', default=0,
                        help='Minimum probability to make predictions for a cell')
    parser.add_argument('-u', '--identifier', default='',
                        help='Unique strain identifier: gene - allele')
    parser.add_argument('-c', '--controls-file', default='',
                        help='Positive and negative controls file')
    parser.add_argument('-x', '--pos-control-cell', default='',
                        help='Positive controls file with single cell labels')
    parser.add_argument('-y', '--pos-control-celldata', default='',
                        help='Positive controls file with single cell labels and data')
    args = parser.parse_args()

    # Neural network hyper-parameters
    param = {'hidden_units': [54, 18],
             'percent_to_test': 0.2,
             'percent_to_valid': 0.2,
             'batch_size': 100,
             'k_fold_cv': 5,
             'learning_rate': 0.01,
             'decay': 1e-6,
             'momentum': 0.9,
             'nesterov': True,
             'num_epochs': 50,
             'runs': 10}

    # Output files
    output_folder = 'ODNN_results'
    if not os.path.isdir(output_folder):
        os.system('mkdir %s' % output_folder)

    output = {'DataScaled': '%s/Data_scaled.csv' % output_folder,
              'PhenotypeCellIDs': '%s/Single_cell_phenotype_predictions.csv' % output_folder,
              'PhenotypeData': '%s/Positive_controls_data.csv' % output_folder,
              'ODresultsWell': '%s/Classification_results_well.csv' % output_folder,
              'ODresultsStrain': '%s/Classification_results_strain.csv' % output_folder,
              'Confusion': '%s/Confusion_matrix.png' % output_folder,
              'TrainingCV': '%s/Training_performance_CV' % output_folder,
              'Training': '%s/Training_performance' % output_folder,
              'CellAccuracy': '%s/Single_cell_accuracies.csv' % output_folder,
              'PCAExplainedVariance': '%s/PCA_explained_variance.txt' % output_folder,
              'PenetranceBins': '%s/Performance_penetrance_bins.csv' % output_folder,
              'ROCPRNumbers': '%s/Performance_ROC_PR.csv' % output_folder}
    # Arguments
    identifier = args.identifier.lower()
    pos_controls_f = [args.controls_f, args.pos_controls_cell_f, args.pos_controls_celldata_f]
    location_feat = ['plate', 'row', 'column']

    # Read input and controls files
    plates, features, mapping, identifiers = read_input_files(args.input_files, args.input_data, args.features_file,
                                                              args.mapping_file, location_feat)
    neg_controls = read_negative_controls_file(args.controls_f, identifier)
    main_dict, dict_feat = initialize_dictionary(identifiers)

    # Read and scale data
    if args.input_data:
        main_dict = read_scaled_data(main_dict, args.input_data, dict_feat, neg_controls, identifier, features)
    else:
        for p in plates:
            main_dict = read_and_scale_plate(main_dict, p, neg_controls, features, mapping,
                                             identifier, identifiers, dict_feat)
        save_data(main_dict, features, identifiers, output)

    # Prepare phenotype data and train NN
    main_dict, phenotype_df, phenotypes = prepare_phenotype_data(main_dict, identifier, identifiers,
                                                                 features, pos_controls_f, output)
    main_dict = make_predictions(main_dict, param, phenotype_df, args.probability, features, output)

    # Save results
    df_well = prepare_output_file_well(main_dict, identifiers, phenotypes, output)
    df_strain = prepare_output_file_strain(main_dict, identifiers, identifier, phenotypes, output)

    # Evaluate performance
    evaluate_performance(args.controls_f, df_well, df_strain, neg_controls, identifier, output)
