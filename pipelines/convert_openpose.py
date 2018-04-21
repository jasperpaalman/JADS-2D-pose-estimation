"""
A pipeline to load all the openpose output
"""
import os
from typing import Sequence

from models import Video
from models.config import Config


def get_videos(config: Config) -> Sequence[Video]:
    for folder in os.listdir(config.openpose_output):

        # Get the sub-folder with openpose data
        open_pose_folder = os.path.join(config.openpose_output, folder)
        # Get the video file
        video_file = os.path.join(config.video_location, folder)

        # use locations and parse to Video object
        if os.path.exists(video_file):
            yield Video.from_open_pose_data(open_pose_folder, video_file)
        # video.get_period_person_division()
        else:
            yield Video.from_open_pose_data(open_pose_folder)

        print('finished video: ', folder)


if __name__ == '__main__':
    # set the local files where the video's and openpose output are stored
    [video.to_json() for video in get_videos(Config.get_config())]
