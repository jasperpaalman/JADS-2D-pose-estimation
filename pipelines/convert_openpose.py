"""
A pipeline to load all the openpose output
"""
import os

from models import Video
from models.config import Config

if __name__ == '__main__':
    # set the local files where the video's and openpose output are stored
    config = Config.get_config()

    # iterate over the folders in the openpose root folder (where there is a subfolder per video)
    for folder in os.listdir(config.openpose_output):

        # Get the sub-folder with openpose data
        open_pose_folder = os.path.join(config.openpose_output, folder)
        # Get the video file
        video_file = os.path.join(config.video_location, folder)

        # use locations and parse to Video object
        if os.path.exists(video_file):
            video: Video = Video.from_open_pose_data(open_pose_folder, video_file)
        else:
            video: Video = Video.from_open_pose_data(open_pose_folder)

        print('finished video: ', folder)
        # Serialise object for later use
        video.to_json()
