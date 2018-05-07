# Run openpose
from typing import Sequence

from learning.regressor import Regressor
from models import Video
from models.config import Config
from models.features import Features
from models.preprocessor import Preprocessor
from pipelines import run_openpose, convert_openpose


def run(do_run_openpose: bool = True,
        do_build_video: bool = True):
    config: Config = Config.get_config()

    # Run openpose
    if do_run_openpose:
        run_openpose.run_openpose(config)

    # convert to readable data types aka period person devision
    videos: Sequence[Video] = None
    if do_build_video:
        videos = convert_openpose.get_videos(config)
        [video.to_json() for video in videos]
    else:
        videos = Video.all_from_json(config.video_data)

    # Convert to usable data type period_running_person division, alle fragment soorten
    preprocessors = list([Preprocessor(video) for video in videos])

    # feature extraction speed / variation / stepping freq
    features = list([Features.from_preprocessor(preprocessor) for preprocessor in preprocessors])

    # build machine learning model
    regressor: Regressor = Regressor(features, 'speed')

    # show output
    print(regressor.evaluate())


if __name__ == '__main__':
    # change if you don't have the data yet
    run(False, False)
