from typing import List, Dict

from models.preprocessor import Preprocessor


class Features:
    """
    TODO implement Features

    TODO implement Serialisation
    """
    def __init__(self, preprocessor: Preprocessor):
        raise NotImplementedError('todo implement @Features.__init__')

    def get_features(self) -> Dict[str, float]:
        raise NotImplementedError('todo implement @Features.get_features')

