from typing import List, Dict

from models.features import Features


class Regressor:
    """
    TODO implement a machine learning solution

    TODO implement Serialisation
    """

    def __init__(self, features_list: List[Features], label: str):
        """
        Constructs (maybe immediately trains the model).
        #todo implement

        :param features_list: The features to train on
        :param label: The feature that should be considered a label
        """

        raise NotImplementedError('Todo implement Regressor')

    def evaluate(self) -> Dict[str, float]:
        """
        Get metrics on how the regressor performed
        #todo implement

        :return: metrics, a dictionary of metric name and value
        """
        raise NotImplementedError('Todo implement Regressor')
