import json


class Config:
    config: 'Config'

    def __init__(self):
        self.__init__()
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
