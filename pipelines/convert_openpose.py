"""
A pipeline to load all the openpose output
"""
import os

from models import Video

if __name__ == '__main__':
    # set the local files where the video's and openpose output are stored
    openpose_output_root = './data/open_pose_output'
    video_root = './data/video'

    # iterate over the folders in the openpose root folder (where there is a subfolder per video)
    for folder in os.listdir(openpose_output_root):

        # Get the subfolder with openpose data
        open_pose_folder = os.path.join(openpose_output_root, folder)
        # Get the video file
        video_file = os.path.join(video_root, folder)

        # use locations and parse to Video object
        if os.path.exists(video_file):
            video: Video = Video.from_open_pose_data(open_pose_folder, video_file)
        else:
            video: Video = Video.from_open_pose_data(open_pose_folder)

        print('finished video: ', folder)
        # Serialise object for later use
        video.to_json()
