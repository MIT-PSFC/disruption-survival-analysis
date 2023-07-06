import numpy as np

from auton_survival.preprocessing import Preprocessor
from auton_survival.estimators import SurvivalModel
from sklearn.ensemble import RandomForestClassifier

class DisruptionPredictor:
    """Analog to how a machine learning model would be seen by the plasma control system
    Data goes in, disruption time comes out
    """

    def __init__(self, name, model, features, transformer:Preprocessor):
        """
        Parameters
        ----------
        name : str
            Name of the predictor
        model : SurvivalModel, or other model
            The model to use for disruption prediction
        features : list of str
            The features used to train the model (should match names in dataset)
        transformer : Preprocessor
            The transformer to use to transform the data before feeding it to the model
            Should match the transformer used to train the model
        """

        self.name = name
        self.model = model
        self.features = features
        self.transformer = transformer

    # Methods for calculating the risk at each time slice for a given shot
    # using different models

    def calculate_risk(self, data, horizon):
        """
        Finds the risk of disruption for a given shot at each time slice
        when looking horizon seconds into the future

        Parameters
        ----------
        data : pandas.DataFrame
            The data to calculate the risk for. 
            Should already be sorted by time and transformed by the predictor's transformer
        horizon : float
            How far into the future the predictor is looking

        Returns
        -------
        risk_time : pandas.DataFrame
            The risk of disruption for each time slice
        """

        raise NotImplementedError("calculate_risk() must be implemented by a subclass of DisruptionPredictor")
    

    # Methods for calculating the disruption time using different algorithms

    def calculate_disruption_time(self, shot_data, threshold, horizon):
        """
        Calculates the disruption time(s) for a given shot with a simple threshold

        Parameters
        ----------
        shot_data : pandas.DataFrame
            Data for a single shot
            Should be sorted by time
            Should be transformed by the predictor's transformer
        threshold : float or list of float
            The threshold(s) to use for determining if a disruption is imminent
            If a list, will return a list of disruption times
        horizon : float
            How far into the future the predictor is looking

        Returns
        -------
        disruption_time : float or list of float
            The time(s) of disruption
            If a list, will return a list of disruption times
            If no disruption is predicted, returns None in that position
        """

        # Calculate the risk for each time slice
        risk_time = self.calculate_risk(shot_data, horizon)

        # Only consider the times more than 'horizon' seconds before the end of the shot
        risk_time = risk_time[risk_time['time'] < shot_data['time'].iloc[-1] - horizon]
        
        # If a single threshold is given, make it a list
        # If a list is given, make a copy of it to keep track of which thresholds have been used
        if isinstance(threshold, np.ndarray):
            avail_thresholds = threshold.tolist()
        else:
            if not isinstance(threshold, list):
                threshold = [threshold]
            avail_thresholds = threshold.copy()

        disruption_times = []
        # Go through the shot data and find the first time the risk exceeds each threshold
        for i in range(len(risk_time)):
            # If there are no more thresholds, stop
            if len(avail_thresholds) == 0:
                break

            # If the risk ever exceeds the threshold, add the time to the list
            # and remove the threshold from the list
            # Then keep going until the risk is below the next threshold
            # or there are no more thresholds
            risk = risk_time.iloc[i]['risk']
            while risk > avail_thresholds[0]:
                disruption_times.append(risk_time.iloc[i]['time'])
                avail_thresholds.pop(0)
                if len(avail_thresholds) == 0:
                    break
        
        # If there is a mismatch between disruption times and thresholds,
        # fill in the rest of the disruption times with None
        if len(disruption_times) < len(threshold):
            for i in range(len(threshold) - len(disruption_times)):
                disruption_times.append(None)

        # If there is only one threshold, return a single disruption time
        if len(disruption_times) == 1:
            return disruption_times[0]
        # Otherwise, return a list of disruption times
        else:
            return disruption_times

    def calculate_disruption_time_hysterisis(self, shot_data, 
                                             upper_threshold, lower_threshold, window,
                                             horizon):
        """
        Calculates the disruption time(s) for a given shot with hysterisis method
        If the 'disruptivity' output of the model goes above the upper threshold
        and remains above the lower threshold for the window length, a disruption
        is predicted
        """
        raise NotImplementedError("Hysterisis method not yet implemented")

class DisruptionPredictorSM(DisruptionPredictor):
    """Disruption Predictors using SurvivalModel from Auton-Survival package"""

    def __init__(self, name, model:SurvivalModel, features, transformer:Preprocessor):
        super().__init__(name, model, features, transformer)

    def calculate_risk(self, data, horizon):

        risk_time = data.copy()

        # Iterate through the data and calculate the risk for each time slice
        try:
            risk_time['risk'] = self.model.predict_risk(data[self.features], horizon)
        except:
            # DSM expects horizons in a list
            risk_time['risk'] = self.model.predict_risk(data[self.features], [horizon])

        # Trim the data to only include the risk and time columns
        return risk_time[['risk', 'time']]
    

class DisruptionPredictorRF(DisruptionPredictor):
    """Disruption Predictors using RandomForestClassifier from sklearn"""

    def __init__(self, name, model:RandomForestClassifier, features, transformer:Preprocessor):
        super().__init__(name, model, features, transformer)

    def calculate_risk(self, data, horizon=None):

        risk_time = data.copy()

        # Iterate through the data and calculate the risk for each time slice
        risk_time['risk'] = self.model.predict_proba(data[self.features])[:,1]

        return risk_time[['risk', 'time']]
    
class DisruptionPredictorTinguely(DisruptionPredictor):
    """Disruption predictor like Tungluely et al. 2019"""

    def __init__(self, name, model:RandomForestClassifier, features, transformer: Preprocessor):
        super().__init__(name, model, features, transformer)

    def linear_slope(self, x, y):
        # Calculate the slope of a linear fit to the data
        # x and y should be numpy arrays
        return (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x**2) - np.sum(x)**2)
    


    def calculate_risk(self, data, horizon):
        # This disruption predictor takes present predicted disruption risk and a
        # linear least-square's fit over some previous time window to predict the
        # disruption risk at some future time
        # P_disrupt(t + horizon) = P_disrupt(t) + slope * horizon
        
        # Time in seconds to do linear fit over
        # In paper, used 0.05, 0.1, 0.2
        t_fit = 0.1 

        risk_time = data.copy()

        # Iterate through the data and calculate the initial risk for each time slice
        risk_time['initial_risk'] = self.model.predict_proba(data[self.features])[:,1]

        # This is a bit of a slow implementation, can speed up later if needed

        fit_times = []
        fit_points = []

        fit_times.append(risk_time.iloc[0]['time'])
        fit_points.append(risk_time.iloc[0]['initial_risk'])

        # Iterate through the data and calculate the slope for each time slice
        # Slope is calculated using a linear fit over the previous t_fit seconds
        for i in range(1, len(risk_time)):

            # Get the time for the new time slice
            new_time = risk_time.iloc[i]['time']

            # Add the new time to the fit
            fit_times.append(new_time)
            fit_points.append(risk_time.iloc[i]['initial_risk'])

            # If the new time is outside the time window, remove the oldest point
            if new_time - fit_times[0] > t_fit:
                fit_times.pop(0)
                fit_points.pop(0)
            
            # Calculate the slope of the linear fit
            slope = self.linear_slope(np.array(fit_times), np.array(fit_points))

            # Extrapolate the risk into the future using the slope
            risk_time.at[i, 'risk'] = risk_time.iloc[i]['initial_risk'] + slope * horizon

        # Replace all NaNs with 0
        risk_time.fillna(0, inplace=True)

        return risk_time[['risk', 'time']]

            






        


