import json
import os
from typing import List, Dict
from data_extraction_methods import get_openpose_output, determine_video_meta_data
from models.config import Config


class Video:
    """
    A class used to more easily get all kinds of data from the openpose output
    Has option to serialise data.

    TODO add type annotation for missing type definitions

    TODO prune irrelevant data types

    TODO split up:
        if some data is enough to stand on its own (for instance all the data used to actually analyse a person
        We should create another data type (e.g. a class Runner) to store and handle that data.
    """
    def __init__(
            self,
            people_per_frame: List[List[Dict]],
            source: str,
            frame_rate: int = -1,
            width: int = -1,
            height: int = -1,
            period_person_division: any = None,
            running_fragments: any = None,
            turning_fragments: any = None,
            fragments: any = None) -> None:
        # TODO find type of period_person and running_fragments
        super().__init__()

        self.people_per_frame = people_per_frame
        self.source = source
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

    def to_json(
            self,
            location: str = None) -> None:
        """
        Serializes to a json file

        :param location: The location to write to.
        """
        if location is None:
            location = os.path.join(Config.get_config().video_data, self.source + ".json")

        json.dump({
            "people_per_frame": self.people_per_frame,
            "frame_rate": self.frame_rate,
            "width": self.width,
            "height": self.height,
            "source": self.source,
        }, open(location, 'w'))

    @staticmethod
    def from_json(
            relative_file_name: str) -> 'Video':
        """
        Retreives a serialised Video object

        :param relative_file_name: The file to retrieve from
        :param folder: The folder to retrieve from.
        :return: The object that is loaded
        """
        data = json.load(open(relative_file_name))

        return Video(
            data['people_per_frame'],
            data['source'],
            data['width'],
            data['height'],
            data['frame_rate']
        )

    @staticmethod
    def all_from_json(folder_name: str = None) -> List['Video']:
        """
        Loads all Video objects stored in folder_name

        :param folder_name: The location to search in
        :return: A list with all the Videos
        """

        if folder_name is None:
            folder_name = Config.get_config().openpose_output

        return [
            Video.from_json(os.path.join(folder_name, file))
            for file in os.listdir(folder_name)
            if file.endswith('.json')
        ]

    @staticmethod
    def from_open_pose_data(
            openpose_folder: str,
            video_location: str = None) -> 'Video':
        """
        Reads a folder with open-pose frames, combines those frames into a Video

        :param openpose_folder: The location of the openpose output.
        :param video_location: The location of the original movie source file
        :return: The video as described by the openpose frames
        """

        source = ''.join(openpose_folder.split('\\')[-1].split('.')[:-1])
        people_per_frame = get_openpose_output(openpose_folder)
        if video_location:
            width, height, frame_rate = determine_video_meta_data(video_location)
            return Video(people_per_frame, source, frame_rate, width, height)
        else:
            return Video(people_per_frame, source)
