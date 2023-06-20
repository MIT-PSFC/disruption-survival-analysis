

from auton_survival.preprocessing import Preprocessor

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

    def calculate_risk(self, data, horizon):
        """
        Finds the risk of disruption for a given shot at each time slice
        when looking horizon seconds into the future

        Parameters
        ----------
        data : pandas.DataFrame
            The data to calculate the risk for
            Should be sorted by time and transformed by the predictor's transformer
        horizon : float
            How far into the future the predictor is looking

        Returns
        -------
        risk_time : pandas.DataFrame
            The risk of disruption for each time slice
        """

        risk_time = data.copy()

        # Iterate through the data and calculate the risk for each time slice
        risk_time['risk'] = self.model.predict_risk(data, horizon)

        return risk_time['risk', 'time']

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
        if not isinstance(threshold, list):
            threshold = [threshold]
        # Make a copy of the thresholds to keep track of which ones have been used
        avail_thresholds = threshold.copy()

        disruption_times = []
        # Go through the shot data and find the first time the risk exceeds each threshold
        for risk in risk_time['risk']:
            # If there are no more thresholds, stop
            if len(avail_thresholds) == 0:
                break

            # If the risk ever exceeds the threshold, add the time to the list
            # and remove the threshold from the list
            # Then keep going until the risk is below the next threshold
            # or there are no more thresholds
            while risk > avail_thresholds[0]:
                disruption_times.append(risk_time['time'])
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

    def _calculate_disruption_time_cph(self):
        pass