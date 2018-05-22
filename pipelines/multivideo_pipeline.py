# Run openpose
from typing import Sequence

from pandas import DataFrame, concat
import time

from models import Video
from models.config import Config
from models.features import Features
from models.preprocessor import Preprocessor
from pipelines import run_openpose, convert_openpose
from functools import reduce


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

    videos = Video.all_from_json(config.video_data)

    # Convert to usable data type period_running_person division, alle fragment soorten
    preprocessors = list([Preprocessor(video) for video in videos])

    # feature extraction speed / variation / stepping freq
    features = list([Features.from_preprocessor(preprocessor) for preprocessor in preprocessors])

    dfs: DataFrame = [feature.feature_df for feature in features]
    final_feature_df = reduce(lambda df1, df2: concat([df1, df2], axis=0, ignore_index=True), dfs)

    final_feature_df.to_csv('final_feature_df {}.csv'.format(time.strftime('%d-%m-%Y')))

if __name__ == '__main__':
    # change if you don't have the data yet
    run(False, False)
