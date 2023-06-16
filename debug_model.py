# For whatever reason, my vscode won't let me step into external libraries
# when running in a jupyter notebook
# debugging in here for the time being

import pandas as pd
import dill
import sys
sys.path.append('../')
from plot_utils import *
from preprocess_datasets import load_features_outcomes, load_features_labels, make_training_sets, get_disruptive_shot_list
from run_models import run_survival_model, run_rf_model, eval_model
from estimators_demo_utils import plot_performance_metrics

# Make training sets if they haven't been created yet

device = 'cmod'
dataset = 'random100'
numeric_feats = ['ip','Wmhd','n_e','kappa','li']

#make_training_sets(device, dataset)

from auton_survival.preprocessing import Preprocessor
# Load and preprocess training, test, validation sets
features_train, outcomes_train = load_features_outcomes(device, dataset+'_train')
features_test, outcomes_test = load_features_outcomes(device, dataset+'_test')
features_val, outcomes_val = load_features_outcomes(device, dataset+'_val')

# The features should match the above
_, labels_train = load_features_labels(device, dataset+'_train', 0.15)
_, labels_test = load_features_labels(device, dataset+'_test', 0.15)
_, labels_val = load_features_labels(device, dataset+'_val', 0.15)

# Fit the imputer and scaler to the training data and transform the training, test, and validation data
preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
transformer = preprocessor.fit(features_train, cat_feats=[], num_feats=numeric_feats, one_hot=True, fill_value=-1)

x_train = transformer.transform(features_train)
x_test = transformer.transform(features_test)
x_val = transformer.transform(features_val)

survival_model_str = 'dsm'

# Run the survival model
#survival_model = run_survival_model(survival_model_str, x_train, x_val, outcomes_train, outcomes_val)

#dill.dump(survival_model, open('models/dsm_model.pkl', 'wb'))

with open ('models/dsm_model.pkl', 'rb') as f:
    survival_model = dill.load(f)

# Evaluate the survival model
#survival_results, survival_times = eval_model(survival_model, x_test, outcomes_train, outcomes_test)

# Plot the results
#survival_title = survival_model_str + ' on ' + dataset + ' dataset'
#plot_performance_metrics(survival_results, survival_times, survival_title)

disruptive_shot = get_disruptive_shot_list(device, dataset+'_test')[0]

plot_risk(device, dataset+'_test', disruptive_shot, 0.15, [survival_model], [survival_model_str], transformer)

"""

# Run the random forest model
rf_model = run_rf_model(x_train, x_val, labels_train, labels_val)

rf_results = rf_model.predict_proba(x_test)[:,1]

# Plot the results
rf_title = 'Random Forest on ' + dataset + ' dataset'
plot_performance_metrics(rf_results, outcomes_test, rf_title)

#plot_survival(device, dataset+'_test', 1160504021, 0.15, [rf_model], transformer)
"""