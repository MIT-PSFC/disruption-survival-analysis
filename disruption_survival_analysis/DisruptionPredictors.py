import numpy as np

from auton_survival.estimators import SurvivalModel
from sklearn.ensemble import RandomForestClassifier

MAX_FUTURE_LIFETIME = 2 # Maximum time to predict into the future (in seconds)
SAMPLE_TIME = 0.001 # Time between samples for integrating survival (in seconds)

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
    
    def get_ettd(self, shot_data):
        """Get the expected time to disruption for each feature vector in shot_data"""

        feature_data = self.get_feature_data(shot_data)
        
        risk_times = np.arange(0.001, MAX_FUTURE_LIFETIME, 0.001)
        # Create chunks of risk times
        risk_times_chunks = np.array_split(risk_times, 10)

        # Split up risk times into intervals and calculate the risk in each interval
        risks_at_horizons = np.zeros((len(shot_data), len(risk_times)))
        for i, chunk in enumerate(risk_times_chunks):
            indices = np.arange(len(chunk)) + i*len(chunk)
            try:
                risks_at_horizons[:, indices] = self.model.predict_risk(feature_data, chunk)
            except:
                # If anything goes wrong, just start appending zeros
                risks_at_horizons[:, indices] = np.zeros((len(chunk), len(shot_data)))

        risk_vals = []
        for i in range(1, len(risk_times)):
            risks_in_interval = risks_at_horizons[:,i] - risks_at_horizons[:,i-1]
            # Replace NaNs with 0
            risks_in_interval = np.nan_to_num(risks_in_interval)
            interval_weight = (risk_times[i] + risk_times[i-1]) / 2
            risk_vals.append(risks_in_interval * interval_weight)

        ettd = np.sum(risk_vals, axis=0)

        # If any values in ettd are negative, set them to 0
        ettd = np.clip(ettd, 0, None)

        return ettd
    
    def get_rmst(self, shot_data):
        """Get the restricted mean survival time for each feature vector in shot_data"""

        feature_data = self.get_feature_data(shot_data)

        integration_times = np.arange(0, MAX_FUTURE_LIFETIME, SAMPLE_TIME)
        # Create chunks of integration times
        integration_times_chunks = np.array_split(integration_times, 10)

        # Split up integration times into intervals and calculate the survival in each interval
        survival_at_horizons = np.zeros((len(shot_data), len(integration_times)))
        for i, chunk in enumerate(integration_times_chunks):
            indices = np.arange(len(chunk)) + i*len(chunk)
            try:
                survival_at_horizons[:, indices] = self.model.predict_survival(feature_data, chunk)
            except:
                # If anything goes wrong, just start appending zeros
                survival_at_horizons[:, indices] = np.zeros((len(chunk), len(shot_data)))


        rmst = np.trapz(survival_at_horizons, integration_times, axis=1)

        return rmst

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
    
    def get_ettd(self, shot_data):
        """Get the expected time to disruption for each feature vector in shot_data"""

        risks = self.get_risks(shot_data)

        # Calculate the expected time to disruption
        # For a random forest (binary classifier) this is (class time / 2) / risk

        ettd = (self.trained_class_time / 2) / risks

        # Limit maximum ettd to be 20x the class time
        ettd = np.clip(ettd, 0, self.trained_class_time * 20)

        return ettd
    
    def get_rmst(self, shot_data):
        """Get the restricted mean survival time for each feature vector in shot_data"""

        # Looking t_horizon into the future, at a sample rate of 1 ms
        risks = self.get_risks(shot_data)

        integration_times = np.arange(0, MAX_FUTURE_LIFETIME, SAMPLE_TIME)

        survival_at_horizons = np.zeros((len(shot_data), len(integration_times)))
        for i, _ in enumerate(integration_times):
            # Probability of surviving into the future at each step is (1 - (risk * t_sample / t_class))
            # So total probability of surviving into the future is the product of all of these
            survival_at_horizons[:,i] = (1 - (risks * SAMPLE_TIME / self.trained_class_time)) ** i
        
        rmst = np.trapz(survival_at_horizons, integration_times, axis=1)
        return rmst
    
class DisruptionPredictorKM(DisruptionPredictor):
    """Kaplan-Meier Disruption predictor like Tinguely et al. 2019"""

    def __init__(self, name, model:RandomForestClassifier, trained_warning_time, trained_class_time, trained_fit_time, trained_horizon):
        super().__init__(name, model, trained_warning_time)
        self.trained_class_time = trained_class_time    # Time before disruption time slices labeled as 'disruptive'
        self.trained_fit_time = trained_fit_time        # Time in seconds to do linear fit over. In paper, used 0.05, 0.1, 0.2
        self.trained_horizon = trained_horizon          # Time in seconds to extrapolate risk into the future

    def calc_slopes(self, probs, times):
        """
        Calculate the extrapolation fit slope for each time slice in shot_data

        Parameters
        ----------
        probs : numpy array
            The output of random forest model for each time slice
        times : numpy array
            The time of each time slice in seconds

        Returns
        -------
        slopes : numpy array
            The slope of the extrapolation fit for each time slice
        """

        # Count how many points are required for the initial fit
        num_fit_points = int(self.trained_fit_time / (times[1] - times[0]))

        slopes = np.zeros(len(probs))
        slopes[0:num_fit_points] = np.nan


        fit_times = times[0:num_fit_points].tolist()
        fit_probs = probs[0:num_fit_points].tolist()

        for i in range(num_fit_points, len(times)):
            # Add the new datapoint to the fit
            fit_times.append(times[i])
            fit_probs.append(probs[i])

            # Remove all points that are outside the fitting window
            while fit_times[-1] - fit_times[0] > self.trained_fit_time:
                fit_times.pop(0)
                fit_probs.pop(0)

            # Create a linear fit for the points
            slope, _ = np.polyfit(fit_times, fit_probs, 1)

            slopes[i] = slope
            
        return slopes

    def get_survival(self, data_times, probs, t_horizon):
        """
        Calculate the probability of not disrupting until t + t_horizon for each feature vector in shot_data
        """

        # Looking t_horizon into the future, at a sample rate of 1 ms
        sample_times = np.arange(0, t_horizon, SAMPLE_TIME)

        # Calculate the slope for each time slice
        slopes = self.calc_slopes(probs, data_times)

        survival_probs = np.zeros(len(probs))

        # Calculate the survival probability for each time slice
        for i, slope in enumerate(slopes):
            if np.isnan(slope):
                survival_probs[i] = np.nan
                continue
            
            # Calculate the probability of not disrupting until t + t_horizon
            P_D = probs[i] + slope * sample_times
            # Restrict P_D to be between 0 and 1
            P_D = np.clip(P_D, 0, 1)
            # Even if the binary classifier predicts 100% we are in the disruptive class
            # the distribution is uniform over the class time so it could survive for a while
            products = 1 - (P_D * SAMPLE_TIME / self.trained_class_time)
            survival_probs[i] = np.prod(products)
                
        return survival_probs

    def get_risks(self, shot_data, t_horizon=None):
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

        if t_horizon is None:
            t_horizon = self.trained_horizon

        data_times = shot_data['time'].values
        probs = self.model.predict_proba(self.get_feature_data(shot_data))[:,1]
        
        # Calculate the probability of not disrupting until t + t_horizon
        survival_probs = self.get_survival(data_times, probs, t_horizon)

        # If any values in survival_probs are NaN, set them to 1
        # This means that anywhere the model predicts NaN for survival (outside fit range), it will predict 0% risk
        survival_probs = np.nan_to_num(survival_probs, nan=1)

        return 1 - survival_probs
    
    def get_rmst(self, shot_data):
        """Get the restricted mean survival time for each feature vector in shot_data"""

        data_times = shot_data['time'].values
        feature_data = self.get_feature_data(shot_data)

        probs = self.model.predict_proba(feature_data)[:,1]

        integration_times = np.arange(0, MAX_FUTURE_LIFETIME, SAMPLE_TIME)

        survival_at_horizons = np.zeros((len(shot_data), len(integration_times)))

        for i, t_horizon in enumerate(integration_times):
            survival_at_horizons[:,i] = self.get_survival(data_times, probs, t_horizon)

        rmst = np.trapz(survival_at_horizons, integration_times, axis=1)

        return rmst
