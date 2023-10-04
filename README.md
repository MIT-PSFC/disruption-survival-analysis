# disruption-survival-analysis

Disruption prediction using [Auton-Survival](https://autonlab.org/auton-survival/)

## Installation
>[!note]
Tested to work with Python 3.10.7 on Windows and Linux

1. Clone this repo
```bash
cd ~/projects
git clone https://github.com/MIT-PSFC/disruption-survival-analysis.git
```

2. Make a new virtual environment and verify using correct python
```bash
cd disruption-survival-analysis
python -m venv .venv
source .venv/Scripts/activate
which python
python --version
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run tests to ensure install is properly working
```bash
python -m unittest tests/test_*
```

## Workflow

### 1. Making Datasets

Datasets used in this repo take the following form:
```
shot, time, time_until_disrupt, feature1, feature2, ...
```

The first three columns are required.
- shot: shot number
- time: measurement time of features (in seconds)
- time_until_disrupt:
	- disruptive shots: Time until disruption
	- non-disruptive shots: NaN

Ordering of the columns and rows makes no difference, library should handle it as long as all the data there.

**To add a new dataset**, follow instructions in `Make Datasets.ipynb`

### 2. Hyperparameter Tuning Models

Follow instructions in `Write Sweep Configs.ipynb` or use `write_sweep_configs.py`

After sweep configs are generated, run the following command to start a hyperparameter tuning session:
```bash
python optuna_job.py models/[device]/[dataset_path]/sweeps/[sweep].yaml
```

Open multiple terminals and execute the command to have several jobs performing sweeps at once. 
- *probably need to re-activate virtual environment for each terminal*

Or, edit job_launch_all.sh to have the correct paths and run the following command:
```bash
chmod +x job_launch_all.sh
./job_launch_all.sh
```

To view results of hyperparameter tuning trials, execute the following command:
```bash
optuna-dashboard sqlite:///models/[device]/[dataset]/studies/[study].db
```

### 3. Running Experiments

Open `Run Experiments.ipynb` notebook

Set the properties of the models that have hyperparameters tuned.

Run all cells, look at graphs.