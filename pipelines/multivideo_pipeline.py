# Run openpose
from typing import Sequence

from learning.regressor import Regressor
from models import Video
from models.config import Config
from models.features import Features
from models.preprocessor import Preprocessor
from pipelines import run_openpose, convert_openpose


def run(do_run_openpose: bool = True):
    config: Config = Config.get_config()

    # Run openpose
    if do_run_openpose:
        run_openpose.run_openpose(config)

    # convert to readable data types aka period person devision
    videos: Sequence[Video] = convert_openpose.get_videos(config)

    # Convert to usable data type period_running_person division, alle fragment soorten
    preprocessors = map(Preprocessor, videos)

    # feature extraction speed / variation / stepping freq
    features = map(Features, preprocessors)

    # build machine learning model
    regressor: Regressor = Regressor(list(features), 'speed')

    # show output
    print(regressor.evaluate())


if __name__ == '__main__':
    # change if you don't have the data yet
    run(False)
