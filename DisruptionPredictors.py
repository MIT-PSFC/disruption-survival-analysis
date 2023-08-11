import numpy as np

from auton_survival.estimators import SurvivalModel
from sklearn.ensemble import RandomForestClassifier

class DisruptionPredictor:
    """Analog to how a machine learning model would be seen by the plasma control system
    Data goes in, disruption time comes out
    This class does NOT store actual data, it only stores the model and features
    """

    def __init__(self, name, model, features, trained_required_warning_time, trained_disruptive_window):
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
        trained_disruptive_window : float
            The disruptive window used to train the model (in seconds)
        """

        self.name = name
        self.model = model
        self.features = features

        self.trained_required_warning_time = trained_required_warning_time
        self.trained_disruptive_window = trained_disruptive_window

    # Methods for calculating the risk at each time slice for a given shot
    # using different models

    def calculate_risk_at_time(self, data, horizon=None):
        """
        Finds the risk of disruption for a given shot at each time slice
        when looking horizon seconds into the future

        Parameters
        ----------
        data : pandas.DataFrame
            The data to calculate the risk for. 
            Should already be sorted by time and transformed
        horizon : float
            How far into the future the predictor is looking

        Returns
        -------
        risk_at_time : pandas.DataFrame
            The risk of disruption for each time slice
        """

        raise NotImplementedError("calculate_risk() must be implemented by a subclass of DisruptionPredictor")
    
    def calculate_ettd_at_time(self, data):
        """
        Calculates the expected time to disruption for a given shot at each time slice.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to calculate the expected time to disruption for
            Should already be sorted by time and transformed
        
        Returns
        -------
        ettd_at_time : pandas.DataFrame
            The expected time to disruption for each time slice
        """
    
        raise NotImplementedError("calculate_ettd_at_time() must be implemented by a subclass of DisruptionPredictor")

class DisruptionPredictorSM(DisruptionPredictor):
    """Disruption Predictors using SurvivalModel from Auton-Survival package"""

    def __init__(self, name, model:SurvivalModel, features, trained_required_warning_time, trained_horizon):
        super().__init__(name, model, features, trained_required_warning_time, trained_horizon)

    def calculate_risk_at_time(self, data, horizon=None):

        if horizon is None:
            horizon = self.trained_disruptive_window

        risk_at_time = data.copy()

        # Iterate through the data and calculate the risk for each time slice
        try:
            risk_at_time['risk'] = self.model.predict_risk(data[self.features], horizon)
        except:
            # DSM expects horizons in a list
            risk_at_time['risk'] = self.model.predict_risk(data[self.features], [horizon])

        # Trim the data to only include the risk and time columns
        return risk_at_time[['risk', 'time']]
    

class DisruptionPredictorRF(DisruptionPredictor):
    """Disruption Predictors using RandomForestClassifier from sklearn"""

    def __init__(self, name, model:RandomForestClassifier, features, trained_required_warning_time, trained_class_time):
        super().__init__(name, model, features, trained_required_warning_time, trained_class_time)

    def calculate_risk_at_time(self, data, horizon=None):

        risk_at_time = data.copy()

        # Iterate through the data and calculate the risk for each time slice
        risk_at_time['risk'] = self.model.predict_proba(data[self.features])[:,1]

        return risk_at_time[['risk', 'time']]
    
class DisruptionPredictorKM(DisruptionPredictor):
    """Kaplan-Meier Disruption predictor like Tinguely et al. 2019"""

    def __init__(self, name, model:RandomForestClassifier, features, trained_warning_time, trained_class_time):
        super().__init__(name, model, features, trained_warning_time, trained_class_time)

    def linear_slope(self, x, y):
        # Calculate the slope of a linear fit to the data
        # x and y should be numpy arrays
        return (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x**2) - np.sum(x)**2)
    
    def calculate_risk_at_time(self, data, horizon=None, t_fit=None):
        # This disruption predictor takes present predicted disruption risk and a
        # linear least-square's fit over some previous time window to predict the
        # disruption risk at some future time
        # P_disrupt(t + horizon) = P_disrupt(t) + slope * horizon
        
        
        if horizon is None:
            horizon = self.trained_disruptive_window

        # Time in seconds to do linear fit over
        # In paper, used 0.05, 0.1, 0.2
        if t_fit is None:
            t_fit = 0.1

        risk_at_time = data.copy()

        # reindex the data to start at 0
        risk_at_time.index = range(len(risk_at_time))

        # Iterate through the data and calculate the initial risk for each time slice
        risk_at_time['initial_risk'] = self.model.predict_proba(data[self.features])[:,1]

        # Initialize risk column to all zeros
        risk_at_time['risk'] = 0

        # This is a bit of a slow implementation, can speed up later if needed

        fit_times = []
        fit_points = []

        fit_times.append(risk_at_time.at[0, 'time'])
        fit_points.append(risk_at_time.at[0, 'initial_risk'])

        # Iterate through the data and calculate the slope for each time slice
        # Slope is calculated using a linear fit over the previous t_fit seconds
        for i in range(1, len(risk_at_time)):

            # Get the time for the new time slice
            new_time = risk_at_time.at[i, 'time']

            # Add the new time to the fit
            fit_times.append(new_time)
            fit_points.append(risk_at_time.at[i, 'initial_risk'])

            # If the new time is outside the time window, remove the oldest point
            if new_time - fit_times[0] > t_fit:
                fit_times.pop(0)
                fit_points.pop(0)
            
            # Calculate the slope of the linear fit
            slope = self.linear_slope(np.array(fit_times), np.array(fit_points))

            # Extrapolate the risk into the future using the slope
            risk_at_time.at[i, 'risk'] = risk_at_time.at[i, 'initial_risk'] + slope * horizon
        
        # Replace all NaNs with the initial risk
        risk_at_time.fillna(risk_at_time['initial_risk'], inplace=True)

        return risk_at_time[['risk', 'time']]

            






        


