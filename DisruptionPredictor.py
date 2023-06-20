

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

        return []

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