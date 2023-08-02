# For whatever reason, my vscode won't let me step into external libraries
# when running in a jupyter notebook
# debugging in here for the time being

from manage_datasets import load_features_outcomes, load_features_labels, make_training_sets, make_stacked_sets, load_feature_list
from model_utils import run_survival_model, run_rf_model, save_model

device = 'cmod'
#dataset_path = 'random_2000_shots_50%_flattop'
dataset_path = 'no_ufo_flattop_1452_shots_50%_disruptive'

# Make training sets if they haven't been created yet
try:
    numeric_feats = load_feature_list(device, dataset_path)
except:
    make_training_sets(device, dataset_path, random_seed=0)
    numeric_feats = load_feature_list(device, dataset_path)

from auton_survival.preprocessing import Preprocessor

# Load and preprocess training, test, validation sets
features_train, outcomes_train = load_features_outcomes(device, dataset_path, 'train', numeric_feats)
features_test, outcomes_test = load_features_outcomes(device, dataset_path, 'test', numeric_feats)
features_val, outcomes_val = load_features_outcomes(device, dataset_path, 'val', numeric_feats)

# Create labels for binary classifier models according to the disruptive window
disruptive_window = 0.15 # seconds before disruption labeled as 'disruptive'
_, labels_train = load_features_labels(device, dataset_path, 'train', disruptive_window, numeric_feats)
_, labels_test = load_features_labels(device, dataset_path, 'test', disruptive_window, numeric_feats)
_, labels_val = load_features_labels(device, dataset_path, 'val', disruptive_window, numeric_feats)

# Fit the imputer and scaler to the training data and transform the training, test, and validation data
preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
transformer=preprocessor.fit(features_train, cat_feats=[], num_feats=numeric_feats, one_hot=True, fill_value=-1)

x_train = transformer.transform(features_train)
x_test = transformer.transform(features_test)
x_val = transformer.transform(features_val)

# Train a dcm model and save it
dcm_model = run_survival_model('dcm', x_train, x_val, outcomes_train, outcomes_val)
save_model(dcm_model, transformer, 'dcm', device, dataset_path, numeric_feats)