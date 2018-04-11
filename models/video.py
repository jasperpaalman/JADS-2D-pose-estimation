import json
import os
from typing import List, Dict
from methods import get_openpose_output, determine_video_meta_data


class Video:
    def __init__(self,
                 people_per_frame: List[List[Dict]],
                 source: str,
                 frame_rate: int = -1,
                 width: int = -1,
                 height: int = -1) -> None:
        super().__init__()

        self.people_per_frame = people_per_frame
        self.source = source
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

    def to_json(self, location: str = None):
        if location is None:
            location = "./data/" + self.source + ".json"

        json.dump({"people_per_frame": self.people_per_frame,
                   "frame_rate": self.frame_rate,
                   "width": self.width,
                   "height": self.height,
                   "source": self.source}, location)

    @staticmethod
    def from_json(relative_file_name: str, folder: str = './data/') -> 'Video':
        data = json.load(os.path.join(folder, relative_file_name))
        return Video(
            data['people_per_frame'],
            data['source'],
            data['width'],
            data['height'],
            data['frame_rate'])

    @staticmethod
    def all_from_json(folder_name: str = 'data/parsed_movies/') -> List['Video']:
        return [
            Video.from_json(os.path.join(folder_name, file))
            for file in os.listdir(folder_name)
            if file.endswith('.json')
        ]

    @staticmethod
    def from_open_pose_data(openpose_folder: str, video_location: str = None) -> 'Video':
        source = ''.join(video_location.split('/')[-1].split('.')[:-1])
        people_per_frame = get_openpose_output(openpose_folder)
        if video_location:
            width, height, frame_rate = determine_video_meta_data(video_location)
            return Video(people_per_frame, source, frame_rate, width, height)
        else:
            return Video(people_per_frame, source)
