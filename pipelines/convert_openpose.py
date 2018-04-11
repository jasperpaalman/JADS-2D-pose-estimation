import os

from models import Video

if __name__ == '__main__':
    root = './data/open_pose_output'
    for folder in os.listdir(root):
        video = Video.from_open_pose_data(os.path.join(root, folder))
        video.to_json()
