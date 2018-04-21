from itertools import chain
from typing import Dict
from numpy import np

import methods
from data_extraction_methods import determine_rmse_threshold, amount_of_frames_to_look_back, rmse
from models import Video


class Preprocessor:
    def __init__(
            self,
            video: Video,
            person_period_division: any = None,
            running_fragments: any = None,
            turning_fragments: any = None,
            fragments: any = None) -> None:
        # TODO find type of period_person and running_fragments
        super().__init__()

        self.source = video.source
        self.frame_rate = video.frame_rate
        self.width = video.width
        self.height = video.height
        self.__period_person_division = self.__get_period_person_division(video)
        self.__person_period_division = person_period_division
        self.__running_fragments = running_fragments
        self.__turning_fragments = turning_fragments
        self.__fragments = fragments

    @staticmethod
    def __get_period_person_division(video: Video) \
            -> Dict[int, Dict[int, any]]:
        """"
        # todo add method description

        :param video: The video object to get the openpose information from.

        :return period_person_division: data structure containing per frame all persons and their
                                       corresponding coordinates
        """
        period_person_division = {}

        # used to create a new person when the algorithm can't find a good person fit based on previous x frames
        next_person = 0

        for frame, file in enumerate(video.people_per_frame):
            # for a frame (period) make a new dictionary in which to store the identified people
            period_person_division[frame] = {}

            for person in file:
                # information for identifying people over disjoint frames
                person_coords = np.array([[x, -y, z] for x, y, z in np.reshape(person['pose_keypoints_2d'], (18, 3))])

                best_person_fit = None  # Initially no best fit person in previous x frames is found
                if frame == 0:  # frame == 0 means no identified people exist (because first frame), so we need to create them ourselves
                    period_person_division[frame][
                        next_person] = person_coords  # create new next people since it is the first frame
                    next_person += 1
                else:
                    # set sufficiently high rmse so it will be overwritten easily
                    min_rmse = 1000

                    # we don't want to base any computation on joints that are not present (==0), so we safe those indices that don't
                    # contain any information
                    empty_joints = set(np.where((person_coords == 0).all(axis=1))[0])

                    # only select used joints
                    used_joints = list(set(range(18)) - empty_joints)
                    # set rmse_threshold equal to the mean distance of each used joint to the center
                    rmse_threshold = determine_rmse_threshold(person_coords, used_joints)

                    # for all possible previous periods within max_frame_diff
                    for i in range(1, amount_of_frames_to_look_back(video.frame_rate, frame) + 1):
                        for earlier_person in period_person_division[frame - i].keys():  # compare with all people
                            if earlier_person not in period_person_division[frame].keys():
                                # if not already contained in current period
                                earlier_person_coords = period_person_division[frame - i][earlier_person]
                                empty_joints_copy = empty_joints.copy()
                                empty_joints_copy = empty_joints_copy | set(
                                    np.where((earlier_person_coords == 0).all(axis=1))[0])
                                used_joints = list(set(range(18)) - empty_joints_copy)
                                if len(used_joints) == 0:
                                    continue
                                # compute root mean squared error based only on mutual used joints
                                person_distance = rmse(earlier_person_coords[used_joints, :],
                                                       person_coords[used_joints, :])

                                # account for rmse threshold (only coordinates very close)
                                if person_distance < rmse_threshold:
                                    if person_distance < min_rmse:  # if best fit, when compared to previous instances
                                        min_rmse = person_distance  # overwrite
                                        best_person_fit = earlier_person  # overwrite
                    if best_person_fit is not None:  # if a best person fit is found
                        period_person_division[frame][best_person_fit] = person_coords
                    else:  # else create new next person
                        period_person_division[frame][next_person] = person_coords
                        next_person += 1
        return period_person_division

    def get_person_period_division(self) \
            -> Dict[int, Dict[int, np.ndarray]]:
        """
        Function that reverses the indexing in the dictionary
        :param period_person_division: data strucure containing per frame all persons and their
                                       corresponding coordinates
        :return person_period_division: data structure containing per person all frames and the coordinates
                                        of that person in that frame.
        """
        person_period_division = {}
        for person in set(chain.from_iterable(self.__period_person_division.values())):
            person_period_division[person] = {}
            for period in self.__period_person_division.keys():
                period_dictionary = self.__period_person_division[period]
                if person in period_dictionary:
                    person_period_division[person][period] = period_dictionary[person]
        return person_period_division

    def get_period_person_division(self) -> any:
        """
        Calculates the private field period_person is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly a period_person is
        """
        if self.__period_person_division is None:
            self.__period_person_division = self.__get_period_person_division(self.people_per_frame, self.frame_rate)

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

        period_running_person_division_df = methods.get_period_running_person_division_df(mean_x_per_person,
                                                                                          moving_people)

        dbscan_subsets = methods.get_dbscan_subsets(maximum_normalized_distance, period_running_person_division_df)
        max_dbscan_subset = dbscan_subsets[
            np.argmax([sum([len(person_period_division[person]) for person in subset]) for subset in dbscan_subsets])]

        running_person_identifiers = methods.determine_running_person_identifiers(period_running_person_division_df,
                                                                                  max_dbscan_subset,
                                                                                  maximum_normalized_distance * 4,
                                                                                  maximum_normalized_distance ** 2)

        self.__running_fragments, self.__turning_fragments, self.__fragments = \
            methods.get_running_and_turning_fragments(running_person_identifiers, mean_x_per_person,
                                                      period_running_person_division_df,
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
