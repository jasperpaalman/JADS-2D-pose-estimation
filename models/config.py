import json


class Config:
    """
    A class used to more easily set all relevant directories for the pipeline
    """
    config: 'Config' = None

    def __init__(self):
        super().__init__()
        data = json.load(open('config.json'))
        self.openpose_output = data['openpose_output']
        self.video_location = data['video_location']
        self.openpose = data['openpose']
        self.video_data = data['video_data']

    @staticmethod
    def get_config() -> 'Config':
        if Config.config is None:
            Config.config = Config()
        return Config.config
