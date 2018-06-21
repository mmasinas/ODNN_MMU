# python ODNN.py -i input/screen1_plates.txt -f input/screen1_feature_set.txt -m input/screens_mapping_sheet.csv -c input/screen1_controls.csv -u strain

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

from ODNN_lib import *
from optparse import OptionParser

# Parameters to specify
parser = OptionParser()
parser.add_option('-i', '--input-files', type='str', dest='input_files',
                  default='', help='Input files to be analyzed')
parser.add_option('-d', '--input-data', type='str', dest='input_data',
                  default='', help='Input data, scaled and filled with identifiers')
parser.add_option('-f', '--features', type='str', dest='features_file',
                  default='', help='Feature Sets to include')
parser.add_option('-m', '--mapping-sheet', type='str', dest='map_file',
                  default='', help='Mapping sheet for strain identifiers')
parser.add_option('-p', '--probability', type='float', dest='probability',
                  default=0, help='Minimum probability to make predictions for a cell')
parser.add_option('-u', '--identifier', type = 'str', dest = 'identifier',
                  default = '', help = 'Unique strain identifier: gene - allele')
parser.add_option('-c', '--control', type='str', dest='control',
                  default='', help='Path to positive and negative controls file')
parser.add_option('-x', '--pos-control-cell', type='str', dest='pos_control_cell',
                  default='', help='Path to positive controls file with single cell labels')
parser.add_option('-y', '--pos-control-celldata', type='str', dest='pos_control_celldata',
                  default='', help='Path to positive controls file with single cell labels and data')
(options, args) = parser.parse_args()

input_data = options.input_data
input_f = options.input_files
features_f = options.features_file
mapping_f = options.map_file
probability = options.probability
identifier = options.identifier.lower()

controls_f = options.control
pos_controls_cell_f = options.pos_control_cell
pos_controls_celldata_f = options.pos_control_celldata
pos_controls_f = [controls_f, pos_controls_cell_f, pos_controls_celldata_f]

# Neural network hyper-parameters
param = {'hidden_units' : [54, 18],
         'percent_to_test': 0.2,
         'percent_to_valid' : 0.2,
         'batch_size': 100,
         'k_fold_cv': 3,
         'learning_rate' : 0.01,
         'decay' : 1e-6,
         'momentum' : 0.9,
         'nesterov' : True,
         'num_epochs' : 50,
         'runs': 2}

# Output files
output_folder = 'ODNN_results'
if not os.path.isdir(output_folder):
    os.system('mkdir %s' % output_folder)
output = {'DataScaled':             '%s/Data_scaled.csv' % output_folder,
          'PhenotypeCellIDs':       '%s/Single_cell_phenotype_predictions.csv' % output_folder,
          'PhenotypeData':          '%s/Positive_controls_data.csv' % output_folder,
          'ODresultsWell':          '%s/Classification_results_well.csv' % output_folder,
          'ODresultsStrain':        '%s/Classification_results_strain.csv' % output_folder,
          'Confusion':              '%s/Confusion_matrix.png' % output_folder,
          'CellAccuracy':           '%s/Single_cell_accuracies.csv' % output_folder,
          'PCAExplainedVariance':   '%s/PCA_explained_variance.txt' % output_folder,
          'PenetranceBins':         '%s/Performance_penetrance_bins.csv' % output_folder,
          'ROCPRNumbers':           '%s/Performance_ROC_PR.csv' % output_folder}

if __name__ == '__main__':

    # Read input and controls files
    location_feat = ['plate', 'row', 'column']
    plates, features, mapping, identifiers = read_input_files(input_f, input_data, features_f, mapping_f, location_feat)
    neg_controls = read_negative_controls_file(controls_f, identifier)
    df, dict_feat = initialize_dictionary(identifiers)

    # Read and scale data
    if input_data:
        df = read_scaled_data(df, input_data, dict_feat, neg_controls, identifier, features)
    else:
        for p in plates:
            df = read_and_scale_plate(df, p, neg_controls, features, mapping, identifier, identifiers, dict_feat)
        save_data(df, features, identifiers, output)

    # Prepare phenotype data and train NN
    df, phenotype_df, phenotypes = prepare_phenotype_data(df, identifier, identifiers, features, pos_controls_f, output)
    df = make_predictions(df, param, phenotype_df, probability, features, output)

    # Save results
    df_OUT = prepare_output_file_well(df, identifiers, phenotypes, output)
    df_OUT_strain = prepare_output_file_strain(df, identifiers, identifier, phenotypes, output)

    # Evaluate performance
    evaluate_performance(controls_f, df_OUT, df_OUT_strain, neg_controls, identifier, output)