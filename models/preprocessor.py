from itertools import chain
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN

from data_extraction_methods import determine_rmse_threshold, amount_of_frames_to_look_back, rmse
from models import Video

from matplotlib import pyplot as plt

class Preprocessor:
    def __init__(
            self,
            video: Video,
            person_period_division: any = None,
            running_fragments: any = None,
            turning_fragments: any = None,
            fragments: any = None,
            running_person_identifiers: any = None) -> None:
        super().__init__()

        if video.frame_rate < 1:
            raise ValueError('video.frame_rate out of bounds "{}"'.format(video.frame_rate))

        self.source = video.source
        self.frame_rate = video.frame_rate
        self.width = video.width
        self.height = video.height
        self.period_person_division = self.get_period_person_division(video)
        self.__moving_people = None
        self.__person_period_division = person_period_division
        self.__running_fragments = running_fragments
        self.__turning_fragments = turning_fragments
        self.__fragments = fragments
        self.__running_person_identifiers = running_person_identifiers
        self.__mean_x_per_person = None

    @staticmethod
    def get_period_person_division(video: Video) \
            -> Dict[int, Dict[int, any]]:
        """
        Uses the openpose output with the aim to relate output across frames to each other, the goal being
        to find coordinates that belong to the same person. This is a step that connects those points and
        groups them under the entity known as a 'person'. In the end the running person is made up of more
        'persons', since the first step that is taken in this function is not yet perfect.

        :param video: The video object to get the openpose information from.

        :return period_person_division: Data structure containing per frame all persons and their
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
                person_coords = np.array([[x, -y, z] for x, y, z in np.reshape(person['pose_keypoints'], (18, 3))])

                best_person_fit = None  # Initially no best fit person in previous x frames is found
                if frame == 0:
                    # frame == 0 means no identified people exist (because first frame),
                    # so we need to create them ourselves
                    period_person_division[frame][
                        next_person] = person_coords  # create new next people since it is the first frame
                    next_person += 1
                else:
                    # set sufficiently high rmse so it will be overwritten easily
                    min_rmse = 1000

                    # we don't want to base any computation on joints that are not present (==0),
                    # so we safe those indices that don't
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
            -> Dict[int, Dict[int, any]]:
        """
        Function that reverses the indexing in the dictionary

        :param period_person_division: Data strucure containing per frame all persons and their
                                       corresponding coordinates

        :return person_period_division: Data structure containing per person all frames and the coordinates
                                        of that person in that frame.
        """
        person_period_division = {}
        for person in set(chain.from_iterable(self.period_person_division.values())):
            person_period_division[person] = {}
            for period in self.period_person_division.keys():
                period_dictionary = self.period_person_division[period]
                if person in period_dictionary:
                    person_period_division[person][period] = period_dictionary[person]
        return person_period_division

    def __set_fragment_sets(self):
        """
        Gets (ready to store) the relevant fragment sets.
        """
        # Get mean_x per person per period (average over all non-empty joints)
        mean_x_per_person = self.get_mean_x_per_person()

        # Get all persons that are moving (exceed movement threshold)
        moving_people = self.get_moving_people()

        # Get a DataFrame with all the information of all moving people
        moving_people_df = self.get_moving_people_df(mean_x_per_person, moving_people)

        # Get the identifiers of all running people
        running_person_identifiers = self.get_running_person_identifiers()

        # Get and set all fragments
        self.__running_fragments, self.__turning_fragments, self.__fragments = \
            Preprocessor.get_running_and_turning_fragments(
                running_person_identifiers,
                mean_x_per_person,
                moving_people_df,
                moving_people, self.frame_rate)

    def get_moving_people(self):
        """
        Get normalized moved distance per person and based on this determine what people are moving in the
        current video

        :return: List with the identifiers of all moving people
        """

        if self.__moving_people is None:
            normalized_moved_distance_per_person = self.get_normalize_moved_distance_per_person()
            maximum_normalized_distance = self.get_maximum_normalized_distance()
            movement_threshold = maximum_normalized_distance / 8
            self.__moving_people = [key for key, value in normalized_moved_distance_per_person.items() if
                                    value > movement_threshold]
        return self.__moving_people

    def get_mean_x_per_person(self) \
            -> Dict[int, Dict[int, float]]:
        """
        Calculate the mean x-position of a person in a certain period

        :param person_period_division: data structure containing per person all frames and the coordinates
                                        of that person in that frame.

        :returns: data structure containing per person all frames and the mean x
                                        of that person in that frame.
        """

        person_period_division = self.get_person_period_division()

        if self.__mean_x_per_person is None:
            self.__mean_x_per_person = {
                person: {period: np.mean(coords[~(coords == 0).any(axis=1), 0])
                         for period, coords in time_coord_dict.items()}
                for person, time_coord_dict in person_period_division.items()
            }

        return self.__mean_x_per_person

    def get_running_fragments(self):
        """
        Calculates the private field running_fragments is None, and returns

        :return: A list of tuples with each tuple containing the start and end frame of a running fragment
        """
        if self.__running_fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__running_fragments

    def get_turning_fragments(self):
        """
        Calculates the private field running_fragments is None, and returns

        :return: A list of tuples with each tuple containing the start and end frame of a turning fragment
        """
        if self.__turning_fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__turning_fragments

    def get_running_person_identifiers(self):
        """
        Function to determine what people are actually running.

        :return: A set containing the person identifiers of running persons
        """
        if self.__running_person_identifiers is None:

            person = self.get_mean_x_per_person()

            mnd = self.get_maximum_normalized_distance()
            moving_people = self.get_moving_people()

            # Use mean x per person and moving people identifiers to create a DataFrame for further analysis
            moving_people_df = \
                self.get_moving_people_df(person, moving_people)

            # Find the DBSCAN clustered set of points that covers the most frames
            max_dbscan_subset = self.get_max_dbscan_subset(mnd, moving_people_df, person)
            max_dbscan_subset = self.get_max_dbscan_subset(mnd, moving_people_df, person)

            # Use the DBSCAN clustered set to determine the other running person identifiers
            self.__running_person_identifiers = \
                self.determine_running_person_identifiers(
                    person, moving_people, max_dbscan_subset, mnd * 4, mnd ** 2)

        return self.__running_person_identifiers

    def get_fragments(self):
        """
        Calculates the private field fragments is None and returns it

        :return: A list of tuples with each tuple containing the start and end frame of a fragment
        """
        if self.__fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__fragments

    def get_maximum_normalized_distance(self):
        """
        :return: maximum normalized moved distance (person entity that moved the fastest)
        """
        return max(self.get_normalize_moved_distance_per_person().values())

    @staticmethod
    def get_moving_people_df(mean_x_per_person: Dict[int, Dict[int, any]],
                                              moving_people: List[int]) -> pd.DataFrame:
        """
        Finding person under observation based on clustering with DBSCAN

        :param mean_x_per_person: Data structure containing per person all frames and the mean x
                                        of that person in that frame.
        :param moving_people: All 'people' that have a normalized moved distance that exceeds a set threshold.
            Contains the running person and possibly noise.

        :return: Dataframe that contains plottable information for all moving people
            (running person + noise)
        """

        return pd.DataFrame(
            [(period, person, x) for person, period_dict in mean_x_per_person.items() if person in moving_people
             for period, x in period_dict.items()], columns=['Period', 'Person', 'X mean'])

    def determine_running_person_identifiers(
            self,
            mean_x_per_person: Dict[int, Dict[int, any]],
            moving_people: List[int],
            max_dbscan_subset: List,
            max_rmse: float,
            max_dist: float) -> any:
        """
        This function takes the largest DBSCAN subset as a starting point and starts expanding to periods that are not
        yet covered.

        For each period not covered yet, the points that are already included are used to create a polynomial
        function to extrapolate from. The points contained within the period are compared and one or zero points can be
        chosen to be included in the main traject/region, which depends on the maximum RMSE that is set.

        If RSME of no point for a period lies below the maximum RMSE, we try again testing for the maximum
        euclidean distance. If no point is included and we move over to the next period in line.
        The periods lower than the initially covered region by DBSCAN is indicated as the lower_periods,
        the periods higher as the upper_periods.

        :param mean_x_per_person: Data structure containing per person all frames and the mean x
                                  of that person in that frame.
        :param moving_people: All 'people' that have a normalized moved distance that exceeds a set threshold.
                              Contains the running person and possibly noise.
        :param max_dbscan_subset: List containing the identifiers to people that are contained in the DBSCAN
                                  subset with the most frames
        :param max_rmse: Threshold under which the RMSE should remain between consecutive points
        :param max_dist: Threshold under which the euclidean distance should remain between consecutive points

        :return running_person_identifiers: The indices of identified 'people' that belong to the running person
        """

        moving_people_df = self.get_moving_people_df(
            mean_x_per_person, moving_people)

        running_person_identifiers = set(max_dbscan_subset)  # set-up plottable people set

        # Make a selection of the dataframe that is contained within the current initial region
        df_sel = moving_people_df[
            moving_people_df['Person'].isin(max_dbscan_subset)].sort_values('Period')

        x = df_sel['Period'].tolist()  # starting x list
        y = df_sel['X mean'].tolist()  # starting y list

        # Region lower and upper bound
        region_lower_bound = \
            moving_people_df[moving_people_df['Person'] == min(max_dbscan_subset)][
                'Period'].min()
        region_upper_bound = \
            moving_people_df[moving_people_df['Person'] == max(max_dbscan_subset)][
                'Period'].max()

        # Determine lower and upper periods to cover
        lower_periods = set(range(moving_people_df['Period'].min(), region_lower_bound)) & set(
            moving_people_df['Period'])
        upper_periods = set(range(region_upper_bound + 1, moving_people_df['Period'].max())) & set(
            moving_people_df['Period'])

        for period in upper_periods:
            x, y, running_person_identifiers = \
                Preprocessor.iterative_main_traject_finder(moving_people_df,
                                                           running_person_identifiers, period, x,
                                                           y,
                                                           max_rmse, max_dist)

        for period in list(lower_periods)[::-1]:
            x, y, running_person_identifiers = \
                Preprocessor.iterative_main_traject_finder(moving_people_df,
                                                           running_person_identifiers, period, x,
                                                           y,
                                                           max_rmse, max_dist)

        return running_person_identifiers

    @staticmethod
    def iterative_main_traject_finder(moving_people_df: pd.DataFrame,
                                      running_person_identifiers: any,
                                      period: int,
                                      x: List,
                                      y: List,
                                      max_rmse: float,
                                      max_dist: float) -> Tuple[List, List, any]:
        """
        Given a period that needs to be tested and some x,y coordinate set to extrapolate from, this function tests,
        based on the maximum RMSE, if the point(s) within this period are comparable with the current region.
        The x,y coordinates are returned as well as the updated plottable people set.


        :param moving_people_df: Dataframe that contains plottable information for all moving people
            (running person + noise)
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param period: Frame that is considered
        :param x: List with all x-coordinates that belong to the running person up until a certain point
        :param y: List with all y-coordinates that belong to the running person up until a certain point
        :param max_rmse: Threshold under which the RMSE should remain between consecutive points
        :param max_dist: Threshold under which the euclidean distance should remain between consecutive points

        :return: Updated versions of x, y and running_person_identifiers
        """

        best_point = None
        dist_point = None

        z = np.polyfit(x, y, 10)  # fit polynomial with sufficient degree to the data points
        f = np.poly1d(z)

        # retrieve values that belong to this period (can contain more than one point, when noise is present)
        period_selection = moving_people_df[moving_people_df['Period'] == period][
            ['Period', 'Person', 'X mean']].values

        # for each of these points check the RMSE
        for period, person, x_mean in period_selection:
            # Calculate RMSE between point and estimation based on the fitted polynomial
            rmse_period = rmse([f(period)], [x_mean])
            if rmse_period < max_rmse:  # First try the RMSE check
                max_rmse = rmse_period
                best_point = (period, x_mean, person)
            # Else try the euclidean distance check
            elif euclidean_distances([[period, x_mean]], list(zip(x, y))).min() < max_dist:
                dist_point = (period, x_mean, person)

        if best_point is not None:  # If a point is found through RMSE
            x.append(best_point[0])
            y.append(best_point[1])
            running_person_identifiers = running_person_identifiers | {int(best_point[2])}

        elif dist_point is not None:  # Else if a point is found through euclidean distance
            x.append(dist_point[0])
            y.append(dist_point[1])
            running_person_identifiers = running_person_identifiers | {int(dist_point[2])}

        # print(period, best_point, dist_point, euclidean_distances([[period, x_mean]], list(zip(x,y))).min())

        return x, y, running_person_identifiers

    def get_normalize_moved_distance_per_person(self) -> Dict[int, float]:
        """
        Calculate moved distance by summing the absolute difference over periods
        Normalize moved distance per identified person over frames by including the average frame difference and the
        length of the number of frames included

        :param mean_x_per_person: Data structure containing per person all frames and the mean x
                                        of that person in that frame.

        :return: A dictionary with as key the person identifier and as value the normalized moved distance
                 by this person (indication of speed)
        """
        mean_x_per_person = self.get_mean_x_per_person()

        normalized_moved_distance_per_person = \
            {
                person: pd.Series(mean_x_dict).diff().abs().sum() / (
                        np.diff(pd.Series(mean_x_dict).index).mean() * len(mean_x_dict)
                )
                for person, mean_x_dict in mean_x_per_person.items()
            }

        # Remove NaN values
        normalized_moved_distance_per_person = {
            key: value for key, value in normalized_moved_distance_per_person.items() if value == value
        }

        return normalized_moved_distance_per_person

    @staticmethod
    def get_max_dbscan_subset(maximum_normalized_distance: float, moving_people_df: pd.DataFrame,
                              person_period_division : Dict[int, Dict[int, any]]):
        """
        Uses a Dataframe with all moving people to cluster points with the DBSCAN clustering method. The goal
        is to find a starting point of clustered points for which we can be fairly sure that it concerns the
        running person (person under observation).

        :param maximum_normalized_distance: Maximum normalized distance (person that is running fastest)
        :param moving_people_df: Dataframe that contains plottable information for all moving people
            (running person + noise)
        :param person_period_division: Data structure containing per person all frames and the coordinates
                                        of that person in that frame.

        :return: List containing the identifiers to people that are contained in the DBSCAN
                                  subset with the most frames
        """
        db = DBSCAN(eps=maximum_normalized_distance, min_samples=1)

        db.fit(moving_people_df[['Period', 'X mean']])

        moving_people_df['labels'] = db.labels_

        dbscan_subsets = moving_people_df.groupby('labels')['Person'].unique().tolist()  # person identifiers by subset

        # Get biggest DBSCAN subset in terms of the amount of frames
        max_dbscan_subset = dbscan_subsets[np.argmax(
            [sum([len(person_period_division[person]) for person in subset]) for subset in dbscan_subsets])]

        return max_dbscan_subset

    @staticmethod
    def get_running_and_turning_fragments(
            running_person_identifiers: List[int],
            mean_x_per_person: Dict,
            moving_people_df,
            moving_people,
            fps: float,
            plot: bool = False):
        """
        Given the identified plottable people (running person/person under observation), this function returns the
        divided running and turning fragments. That is, each running fragment is a person running from one side to the other
        and the turning fragments are the fragments that remain.

        :param fps: The frame rate of the video
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param mean_x_per_person: Mean x-position of a person in a certain period (average over all joints)
        :param moving_people_df: Dataframe that contains plottable information for all moving people
            (running person + noise)
        :param moving_people: All 'people' that have a normalized moved distance that exceeds a set threshold.
            Contains the running person and possibly noise.

        :returns running_fragments, turning_fragments, fragments:
            All are a list of tuples. Each tuple indicates the start frame and end frame of a fragment.
            Running fragments indicate the estimated fragments where the person under observation is running
            Turning fragments indicate the estimated fragments where the person is either slowing down, turning or starting,
            i.e. not solely running
        """

        if plot:
            # Plot the original dataframe to show the difference between moving_people (incl. noise)
            # and the extract running_person_identifiers
            pd.DataFrame({key: value for key, value in mean_x_per_person.items() if key in moving_people}).plot()

        # Retrieve dataframe, but only select running person identifiers
        running_person_identifiers_df = moving_people_df[
            moving_people_df['Person'].isin(running_person_identifiers)].sort_values(
            'Period')

        # Get frames as x values and x-mean as y values
        x = running_person_identifiers_df['Period'].values
        y = running_person_identifiers_df['X mean'].values

        min_period = running_person_identifiers_df['Period'].min()  # minimum period/frame
        max_period = running_person_identifiers_df['Period'].max()  # maximum period/frame

        # fit polynomial with sufficient degree to the data points
        z = np.polyfit(x, y, 20)
        f = np.poly1d(z)

        # Construct new smooth line using the polynomial function
        # Number of points are the number of periods times a multiplication factor
        xnew = np.linspace(min_period, max_period, num=len(x) * 10, endpoint=True)
        ynew = f(xnew)

        # Determine optima indexes for xnew and ynew with a function checking if there is a sign change
        optima_ix = np.where(np.diff(np.sign(np.diff(ynew))) != 0)[0] + 1

        # The optima reflect the turning points, which can be retrieved in terms of frames
        frame_optima = xnew[optima_ix]
        frame_optima = list(
            map(lambda t: min(running_person_identifiers_df['Period'].unique(), key=lambda x: abs(x - t)),
                frame_optima))

        # Add minimum and maximum period/frame of the interval we look at
        frame_optima = sorted(set(frame_optima) | set([min_period, max_period]))

        # Locate the x-mean values that belong to these points
        z = np.polyfit(x, y, 10)
        f = np.poly1d(z)
        x_optima = list(f(frame_optima))

        # Find the relevant points by checking if it is the minimum and maximum period/frame of the interval or
        # if the points are sufficiently apart in both x and y coordinate
        points_x = []
        points_y = []
        for frame, x_mean in zip(frame_optima, x_optima):
            if frame in [min_period, max_period]:  # always add first and last frame
                points_x.append(frame)
                points_y.append(x_mean)
            # Add next identified frame only if the x_mean difference is larger than a threshold
            elif abs(points_y[-1] - x_mean) > 300:
                points_x.append(frame)
                points_y.append(x_mean)

        # Derive fragments also with last check if frames indicating the start/end of a fragment are
        # sufficiently far apart
        fragments = [(i, j) for i, j in zip(points_x, points_x[1:]) if j - i > 2*fps]

        if plot:
            # Plot found information
            plt.plot(xnew, ynew)
            plt.plot(points_x, points_y, "o", color='orange')
            plt.title('Finding turning points (optima) and deriving fragments')
            plt.xlabel('Frames')
            plt.ylabel('X position')
            plt.legend().set_visible(False)

            # Plot coordinates that will be used in further analyses + identified slow down points
            pd.DataFrame(
                {key: value for key, value in mean_x_per_person.items() if key in running_person_identifiers}).plot()
            plt.title(
                'All coordinates that will be used in further analyses & \n identified points where a start or a slow-down is finalized')
            plt.xlabel('Frames')
            plt.ylabel('X position')
            plt.legend().set_visible(False)

        ### Splitting turning and running motions ###

        running_person_identifiers_df.set_index('Period')

        turning_frames = []

        # Individually look at each fragment (this contains parts that are a running motion and parts that are a turning/starting motion)
        # Rationale: Check for maximum change in accelertion by taking minimum/maximum of second derivative

        avg_first_derivative = np.mean(abs(np.diff(ynew)))  # average first derivative
        avg_second_derivative = np.mean(abs(np.diff(np.diff(ynew))))  # average second derivative

        for fragment in fragments:
            # mask for selecting the coordinates of each fragment
            mask = (xnew >= fragment[0]) & (xnew < fragment[1])

            # selection of frames (x_sel) and x-position (y_sel)
            x_sel = xnew[mask]
            y_sel = ynew[mask]

            # turn into series
            fragment_series = pd.Series(y_sel, index=x_sel)

            periods = fragment_series.index.tolist()

            # We split the fragment in a x number of parts and look only at the middle part (i.e. interval [1/x, x-1/x] )
            # In this interval we will find the deviations in both sides from second derivative == 0 and mark those as the start and end point of a running fragment
            split = 5
            lower_bound = periods[len(periods) // split]
            upper_bound = periods[(len(periods) // split) * (split - 1)]

            secondDerivative = fragment_series.diff().diff()  # calculating second derivative

            # At second derivative == 0, a person reaches his/her top speed
            # Around this point we set a certain confidence bound, for which we can be fairly sure that a person is
            # running
            # Since the model tended to include slowing down a lot, a penalty is added relative to the slowing down
            # The map function makes sure that when no value falls within the interval that the upper or lower bound is
            # assigned

            start_up_thresh = avg_second_derivative
            slow_down_thresh = (2 / 5) * start_up_thresh

            if y_sel[0] < y_sel[-1]:
                min_frame = secondDerivative[(fragment_series.diff().abs() > avg_first_derivative) & (
                    secondDerivative.apply(lambda x: 0 < x < start_up_thresh))].loc[lower_bound:].index.min()
                min_frame = list(map(lambda x: lower_bound if x != x else x, [min_frame]))[0]

                max_frame = secondDerivative[(fragment_series.diff().abs() > avg_first_derivative) & (
                    secondDerivative.apply(lambda x: -slow_down_thresh < x < 0))].loc[:upper_bound].index.max()
                max_frame = list(map(lambda x: upper_bound if x != x else x, [max_frame]))[0]

                turning_frames.append(min_frame)
                turning_frames.append(max_frame)
            else:
                min_frame = secondDerivative[(fragment_series.diff().abs() > avg_first_derivative) & (
                    secondDerivative.apply(lambda x: -start_up_thresh < x < 0))].loc[lower_bound:].index.min()
                min_frame = list(map(lambda x: lower_bound if x != x else x, [min_frame]))[0]

                max_frame = secondDerivative[(fragment_series.diff().abs() > avg_first_derivative) & (
                    secondDerivative.apply(lambda x: 0 < x < slow_down_thresh))].loc[:upper_bound].index.max()
                max_frame = list(map(lambda x: upper_bound if x != x else x, [max_frame]))[0]

                turning_frames.append(min_frame)
                turning_frames.append(max_frame)

        if plot:
            # get the estimated x-position where a start or a slow-down is finalized (just for plotting purposes)
            z = np.polyfit(x, y, 10)
            f = np.poly1d(z)
            turning_x = f(turning_frames)

            # Adding points to plot where a start or a slow-down is finalized (just for plotting purposes)
            plt.scatter(turning_frames, turning_x)

        all_points = sorted(set(turning_frames) | {min_period, max_period})
        all_fragments = list(zip(all_points, all_points[1:]))

        # Splitting running fragments and turning fragments
        running_fragments = all_fragments[1::2]
        turning_fragments = all_fragments[::2]

        return running_fragments, turning_fragments, fragments
