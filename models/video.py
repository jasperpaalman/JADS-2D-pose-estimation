import json
import os
from typing import List, Dict
import methods
from models.config import Config
import numpy as np


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
        self.__period_person_division = period_person_division
        self.__running_fragments = running_fragments
        self.__turning_fragments = turning_fragments
        self.__fragments = fragments

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
            "period_person_division": self.__period_person_division,
            "running_fragments": self.__running_fragments,
            "turning_fragments": self.__turning_fragments,
            "fragments": self.__fragments
        }, open(location, 'w'))

    @staticmethod
    def from_json(
            relative_file_name: str,
            folder: str = None) -> 'Video':
        """
        Retreives a serialised Video object

        :param relative_file_name: The file to retrieve from
        :param folder: The folder to retrieve from.
        :return: The object that is loaded
        """
        data = json.load(os.path.join(folder, relative_file_name))
        if folder is None:
            folder = Config.get_config().video_data

        return Video(
            data['people_per_frame'],
            data['source'],
            data['width'],
            data['height'],
            data['frame_rate'],
            data['period_person_division'],
            data['running_fragments'],
            data['turning_fragments'],
            data['fragments']
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
        people_per_frame = methods.get_openpose_output(openpose_folder)
        if video_location:
            width, height, frame_rate = methods.determine_video_meta_data(video_location)
            return Video(people_per_frame, source, frame_rate, width, height)
        else:
            return Video(people_per_frame, source)

    def get_period_person_division(self) -> any:
        """
        Calculates the private field period_person is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly a period_person is
        """
        if self.__period_person_division is None:
            self.__period_person_division = methods.get_period_person_division(self.people_per_frame, self.frame_rate)

        # do logic to set self.__period_person
        return self.__period_person_division

    def __set_fragment_sets(self):
        """
        Gets (ready to store) the relevant fragment sets.
        Copied from Pipeline_new.ipynb
        # todo cleanup work
        """
        person_period_division = self.get_period_person_division()

        mean_x_per_person = methods.get_mean_x_per_person(person_period_division)

        normalized_moved_distance_per_person = methods.normalize_moved_distance_per_person(mean_x_per_person)

        # Only include identified people that move more than a set movement threshold
        maximum_normalized_distance = max(normalized_moved_distance_per_person.values())
        movement_threshold = maximum_normalized_distance / 4
        moving_people = [key for key, value in normalized_moved_distance_per_person.items() if
                         value > movement_threshold]

        person_plottables_df = methods.get_person_plottables_df(mean_x_per_person, moving_people)

        dbscan_subsets = methods.get_dbscan_subsets(maximum_normalized_distance, person_plottables_df)
        max_dbscan_subset = dbscan_subsets[
            np.argmax([sum([len(person_period_division[person]) for person in subset]) for subset in dbscan_subsets])]

        plottable_people = methods.determine_plottable_people(person_plottables_df,
                                                              max_dbscan_subset,
                                                              maximum_normalized_distance * 4,
                                                              maximum_normalized_distance ** 2)

        self.__running_fragments, self.__turning_fragments, self.__fragments = \
            methods.get_running_and_turning_fragments(plottable_people, mean_x_per_person, person_plottables_df,
                                                      moving_people, self.frame_rate)

    def get_running_fragments(self):
        """
        Calculates the private field running_fragments is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly running_fragments are
        """
        if self.__running_fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__running_fragments

    def get_turning_fragments(self):
        """
        Calculates the private field running_fragments is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly turning_fragments are
        """
        if self.__turning_fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__turning_fragments

    def fragments(self):
        """
        Calculates the private field fragments is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly fragments are
        """
        if self.__fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__fragments
