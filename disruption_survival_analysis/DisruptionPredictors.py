import numpy as np

from auton_survival.estimators import SurvivalModel
from sklearn.ensemble import RandomForestClassifier

class DisruptionPredictor:
    """Analog to how a machine learning model would be seen by the plasma control system
    Data goes in, risk or expected time to disruption comes out
    """

    def __init__(self, name, model, trained_required_warning_time):
        """
        Parameters
        ----------
        name : str
            Name of the predictor
        model : SurvivalModel, or other model
            The model to use for disruption prediction
        features : list of str
            The features used to train the model (should match names in dataset)
        trained_required_warning_time : float
            The required warning time used to train the model (in seconds)
        trained_class_time: float, optional
            The class time used to train binary classifier part of model (in seconds)
        trained_horizon : float, optional
            The horizon used to train the model (in seconds)
        """

        self.name = name
        self.model = model

        self.trained_required_warning_time = trained_required_warning_time

    def get_feature_data(self, shot_data):
        """Get the feature data from the shot data"""
        feature_names = list(shot_data.columns)
        feature_names.remove('shot')
        feature_names.remove('time')
        feature_names.remove('time_until_disrupt')

        feature_data = shot_data[feature_names]

        return feature_data

class DisruptionPredictorSM(DisruptionPredictor):
    """Disruption Predictors using SurvivalModel from Auton-Survival package"""

    def __init__(self, name, model:SurvivalModel, trained_required_warning_time, trained_horizon):
        super().__init__(name, model, trained_required_warning_time)

        self.trained_horizon = trained_horizon

    def get_risks(self, shot_data, horizon=None):
        """
        Calculate the risk of disruption for each feature vector in shot_data

        Parameters
        ----------
        shot_data : pandas.DataFrame
            The data to calculate the risk for. Expects the data to include features the model was trained on.
        horizon : float, optional
            The horizon to calculate the risk for (in seconds). If None, use the horizon the model was trained on.
        
        Returns
        -------
        risk : numpy array
            The risk of disruption for each feature vector in shot_data
        """

        if horizon is None:
            horizon = self.trained_horizon

        feature_data = self.get_feature_data(shot_data)

        try:
            risks = self.model.predict_risk(feature_data, horizon)
        except:
            # DSM expects horizons in a list
            risks = self.model.predict_risk(feature_data, [horizon])

        return risks[:,0]

class DisruptionPredictorRF(DisruptionPredictor):
    """Disruption Predictors using RandomForestClassifier from sklearn"""

    def __init__(self, name, model:RandomForestClassifier, trained_required_warning_time, trained_class_time):
        super().__init__(name, model, trained_required_warning_time)

        self.trained_class_time = trained_class_time

    def get_risks(self, shot_data):
        """
        Calculate the risk of disruption for each feature vector in shot_data

        Parameters
        ----------
        shot_data : pandas.DataFrame
            The data to calculate the risk for. Expects the data to include features the model was trained on.
        
        Returns
        -------
        risk : numpy array
            The risk of disruption for each feature vector in shot_data
        """

        feature_data = self.get_feature_data(shot_data)

        risks = self.model.predict_proba(feature_data)[:,1]

        return risks
    
class DisruptionPredictorKM(DisruptionPredictor):
    """Kaplan-Meier Disruption predictor like Tinguely et al. 2019"""

    def __init__(self, name, model:RandomForestClassifier, trained_warning_time, trained_class_time, trained_fit_time, trained_horizon):
        super().__init__(name, model, trained_warning_time)
        self.trained_class_time = trained_class_time    # Time before disruption time slices labeled as 'disruptive'
        self.trained_fit_time = trained_fit_time        # Time in seconds to do linear fit over. In paper, used 0.05, 0.1, 0.2
        self.trained_horizon = trained_horizon          # Time in seconds to extrapolate risk into the future

    def get_risks(self, shot_data, horizon=None):
        """
        Calculate the risk of disruption for each feature vector in shot_data

        Parameters
        ----------
        shot_data : pandas.DataFrame
            The data to calculate the risk for. Expects the data to include features the model was trained on.
        horizon : float
            The horizon to calculate the risk for (in seconds).
        
        Returns
        -------
        risks : numpy array
            The risk of disruption for each feature vector in shot_data
        """

        # This disruption predictor takes present predicted disruption risk and a
        # linear least-square's fit over some previous time window to predict the
        # disruption risk at some future time
        # P_disrupt(t + horizon) = intercept + slope * (t + horizon)
        
        times = shot_data['time'].values
        feature_data = self.get_feature_data(shot_data)

        initial_risks = self.model.predict_proba(feature_data)[:,1]

        # This is a bit of a slow implementation, can speed up later if needed

        fit_times = [times[0]]
        fit_points = [initial_risks[0]]

        risks = np.zeros(len(initial_risks))

        # Iterate through the data and calculate the slope for each time slice
        # Slope is calculated using a linear fit over the previous t_fit seconds
        for i in range(1, len(initial_risks)):

            # Get the time for the new time slice
            new_time = times[i]

            # Add the new datapoint to the fit
            fit_times.append(new_time)
            fit_points.append(initial_risks[i])

            # Remove all points that are outside the fitting window
            while fit_times[-1] - fit_times[0] > self.trained_fit_time:
                fit_times.pop(0)
                fit_points.pop(0)
            
            # Create a linear fit for the points
            slope, intercept = np.polyfit(fit_times, fit_points, 1)

            # Extrapolate the risk into the future using this line
            risks[i] = intercept + (slope * (times[i] + horizon))
        
        # Replace all NaNs with the initial risk
        risks = np.nan_to_num(risks, nan=initial_risks)

        return risks[:,0]




            






        


