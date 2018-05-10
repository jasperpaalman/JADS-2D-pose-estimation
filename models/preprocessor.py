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
        # TODO find type of period_person and running_fragments
        super().__init__()

        if video.frame_rate < 1:
            raise ValueError('video.frame_rate out of bounds "{}"'.format(video.frame_rate))

        self.source = video.source
        self.frame_rate = video.frame_rate
        self.width = video.width
        self.height = video.height
        self.__moving_people = None
        self.__period_person_division = self.__get_period_person_division(video)
        self.__person_period_division = person_period_division
        self.__running_fragments = running_fragments
        self.__turning_fragments = turning_fragments
        self.__fragments = fragments
        self.__running_person_identifiers = running_person_identifiers
        self.__mean_x_per_person = None

        # Unserialised

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
        mean_x_per_person = self.get_mean_x_per_person()

        moving_people = self.get_moving_people()

        period_running_person_division_df = self.get_period_running_person_division_df(mean_x_per_person,
                                                                                       moving_people)

        running_person_identifiers = self.get_running_person_identifiers()

        self.__running_fragments, self.__turning_fragments, self.__fragments = \
            Preprocessor.get_running_and_turning_fragments(
                running_person_identifiers,
                mean_x_per_person,
                period_running_person_division_df,
                moving_people, self.frame_rate)

    def get_moving_people(self):
        if self.__moving_people is None:
            normalized_moved_distance_per_person = self.get_normalize_moved_distance_per_person()
            maximum_normalized_distance = self.get_maximum_normalized_distance()
            movement_threshold = maximum_normalized_distance / 4
            self.__moving_people = [key for key, value in normalized_moved_distance_per_person.items() if
                                    value > movement_threshold]
        return self.__moving_people

    def get_mean_x_per_person(self) \
            -> Dict[int, Dict[int, float]]:
        """
        Calculate the mean x-position of a person in a certain period

        :param person_period_division:
        :returns: a dictionary
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

    def get_running_person_identifiers(self):
        if self.__running_person_identifiers is None:
            # TODO set params
            person = self.get_mean_x_per_person()

            mnd = self.get_maximum_normalized_distance()
            moving_people = self.get_moving_people()

            period_running_person_division_df = \
                self.get_period_running_person_division_df(person, moving_people)

            max_dbscan_subset = self.get_max_dbscan_subset(mnd, period_running_person_division_df, person)

            self.__running_person_identifiers = \
                self.determine_running_person_identifiers(
                    person, moving_people, max_dbscan_subset, mnd * 4, mnd ** 2)

        return self.__running_person_identifiers

    def get_fragments(self):
        """
        Calculates the private field fragments is None, and returns
        # todo: specity return type

        :return: #todo specify what exactly fragments are
        """
        if self.__fragments is None:
            self.__set_fragment_sets()

        # do logic to set self.__period_person
        return self.__fragments

    def get_maximum_normalized_distance(self):
        return max(self.get_normalize_moved_distance_per_person().values())

    @staticmethod
    def get_period_running_person_division_df(mean_x_per_person: Dict[int, Dict[int, any]],
                                              moving_people: List[int]) -> pd.DataFrame:
        """
        Finding person under observation based on clustering with DBSCAN

        :param mean_x_per_person:
        :param moving_people:
        :return:
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

        If RSME of no point for a period lies below the maximum RMSE,
        no point is included and we move over to the next period in line
        The periods lower than the initially covered region by DBSCAN is indicated as the lower_periods,
        the periods higher as the upper_periods.

        TODO: Define type of max_dbscan_subset elements, define use of variables.

        :param moving_people:
        :param mean_x_per_person:
        :param max_dist:
        :param max_rmse:
        :param max_dbscan_subset:
        :param period_running_person_division_df:

        :return running_person_identifiers:
        """

        period_running_person_division_df = self.get_period_running_person_division_df(
            mean_x_per_person, moving_people)

        running_person_identifiers = set(max_dbscan_subset)  # set-up plottable people set

        # Make a selection of the dataframe that is contained within the current initial region
        df_sel = period_running_person_division_df[
            period_running_person_division_df['Person'].isin(max_dbscan_subset)].sort_values('Period')

        x = df_sel['Period'].tolist()  # starting x list
        y = df_sel['X mean'].tolist()  # starting y list

        # Region lower and upper bound
        region_lower_bound = \
            period_running_person_division_df[period_running_person_division_df['Person'] == min(max_dbscan_subset)][
                'Period'].min()
        region_upper_bound = \
            period_running_person_division_df[period_running_person_division_df['Person'] == max(max_dbscan_subset)][
                'Period'].max()

        # Determine lower and upper periods to cover
        lower_periods = set(range(period_running_person_division_df['Period'].min(), region_lower_bound)) & set(
            period_running_person_division_df['Period'])
        upper_periods = set(range(region_upper_bound + 1, period_running_person_division_df['Period'].max())) & set(
            period_running_person_division_df['Period'])

        for period in upper_periods:
            x, y, running_person_identifiers = \
                Preprocessor.iterative_main_traject_finder(period_running_person_division_df,
                                                           running_person_identifiers, period, x,
                                                           y,
                                                           max_rmse, max_dist)

        for period in list(lower_periods)[::-1]:
            x, y, running_person_identifiers = \
                Preprocessor.iterative_main_traject_finder(period_running_person_division_df,
                                                           running_person_identifiers, period, x,
                                                           y,
                                                           max_rmse, max_dist)

        return running_person_identifiers

    def iterative_main_traject_finder(period_running_person_division_df: pd.DataFrame,
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

        TODO: improve type notation for x, y and define purpose of parameters (define type of plottable people)

        :param max_dist:
        :param period_running_person_division_df:
        :param running_person_identifiers:
        :param period:
        :param x:
        :param y:
        :param max_rmse:
        :return:
        """

        best_point = None
        dist_point = None

        z = np.polyfit(x, y, 10)  # fit polynomial with sufficient degree to the datapoints
        f = np.poly1d(z)

        # retrieve values that belong to this period (can contain more than one point, when noise is present)
        period_selection = period_running_person_division_df[period_running_person_division_df['Period'] == period][
            ['Period', 'Person', 'X mean']].values

        # for each of these points check the RMSE
        for period, person, x_mean in period_selection:
            rmse_period = rmse([f(period)], [x_mean])
            if rmse_period < max_rmse:
                max_rmse = rmse_period
                best_point = (period, x_mean, person)
            elif euclidean_distances([[period, x_mean]], list(zip(x, y))).min() < max_dist:
                dist_point = (period, x_mean, person)

        if best_point is not None:
            x.append(best_point[0])
            y.append(best_point[1])
            running_person_identifiers = running_person_identifiers | {int(best_point[2])}

        elif dist_point is not None:
            x.append(dist_point[0])
            y.append(dist_point[1])
            running_person_identifiers = running_person_identifiers | {int(dist_point[2])}

        # print(period, best_point, dist_point, euclidean_distances([[period, x_mean]], list(zip(x,y))).min())

        return x, y, running_person_identifiers

    def get_normalize_moved_distance_per_person(self) -> Dict[int, int]:
        """
        Calculate moved distance by summing the absolute difference over periods
        Normalize moved distance per identified person over frames by including the average frame difference and the length
        of the number of frames included

        :param mean_x_per_person: A Persons containing their frames containing the mean x for that person for that frame.
        :return:
        """
        mean_x_per_person = self.get_mean_x_per_person()

        normalized_moved_distance_per_person = \
            {
                person: pd.Series(mean_x_dict).diff().abs().sum() / (
                        np.diff(pd.Series(mean_x_dict).index).mean() * len(mean_x_dict)
                )
                for person, mean_x_dict in mean_x_per_person.items()
            }

        normalized_moved_distance_per_person = {
            key: value for key, value in normalized_moved_distance_per_person.items() if value == value
        }
        return normalized_moved_distance_per_person

    @staticmethod
    def get_max_dbscan_subset(maximum_normalized_distance: float, period_running_person_division_df: pd.DataFrame,
                              person_period_division : Dict[int, Dict[int, any]]):
        """

        :param maximum_normalized_distance:
        :param period_running_person_division_df:
        :return:
        """
        db = DBSCAN(eps=maximum_normalized_distance, min_samples=1)

        db.fit(period_running_person_division_df[['Period', 'X mean']])

        period_running_person_division_df['labels'] = db.labels_

        # maximum_label = period_running_person_division_df.groupby('labels').apply(len).sort_values(ascending=False).index[0]

        dbscan_subsets = period_running_person_division_df.groupby('labels')['Person'].unique().tolist()

        max_dbscan_subset = dbscan_subsets[np.argmax(
            [sum([len(person_period_division[person]) for person in subset]) for subset in dbscan_subsets])]

        return max_dbscan_subset

    @staticmethod
    def get_running_and_turning_fragments(
            running_person_identifiers: List[int],
            mean_x_per_person: Dict,
            period_running_person_division_df,
            moving_people,
            fps: float,
            plot: bool = False):
        """
        TODO: Needs some love this method is waaaaaay too long.

        Given the identified plottable people (running person/person under observation), this function returns the
        divided running and turning fragments. That is, each running fragment is a person running from one side to the other
        and the turning fragments are the fragments that remain.

        :param fps: The frame rate of the video
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param mean_x_per_person: Mean x-position of a person in a certain period (average over all joints)
        :param period_running_person_division_df: Dataframe that contains plottable information for all moving people
            (running person + noise)
        :param moving_people: All 'people' that have a normalized moved distance that exceeds a set threshold.
            Contains the running person and possibly noise.

        :returns running_fragments, turning_fragments: Both are a list of tuples. Each tuple indicated the start frame and
            end frame of a fragment.
            Running fragments indicate the estimated fragments where the person under observation is running
            Turning fragments indicate the estimated fragments where the person is either slowing down, turning or starting,
            i.e. not solely running
        """

        # Plot the original dataframe to show the difference between moving_people (incl. noise)
        # and the extract running_person_identifiers
        pd.DataFrame({key: value for key, value in mean_x_per_person.items() if key in moving_people}).plot()

        # Retrieve dataframe, but only select plottable people
        running_person_identifiers_df = period_running_person_division_df[
            period_running_person_division_df['Person'].isin(running_person_identifiers)].sort_values(
            'Period')

        x = running_person_identifiers_df['Period'].values
        y = running_person_identifiers_df['X mean'].values

        min_period = running_person_identifiers_df['Period'].min()  # minimum period/frame
        max_period = running_person_identifiers_df['Period'].max()  # maximum period/frame

        # fit polynomial with sufficient degree to the datapoints
        z = np.polyfit(x, y, 20)
        f = np.poly1d(z)

        # Construct new smooth line using the polynomial function
        # Number of points are the number of periods times a multiplication factor
        xnew = np.linspace(min_period, max_period, num=len(x) * 10, endpoint=True)
        ynew = f(xnew)

        # Determine optima indexes for xnew and ynew
        # Function checks if there is a sign change and if sufficient points (# indicated through 'periods' variable)
        # surrounding this candidate turning point are not changing in sign
        periods = int(fps * 10)
        periods_sign_diff = np.diff(np.sign(np.diff(ynew)))
        optima_ix = [i + 1 for i in range(len(periods_sign_diff)) if periods_sign_diff[i] != 0
                     and (periods_sign_diff[i - periods:i] == 0).all()
                     and (periods_sign_diff[i + 1:i + periods + 1] == 0).all()]  # local min+max

        # The optima reflect the turning points, which can be retrieved in terms of frames
        turning_points = xnew[optima_ix]
        turning_points = list(
            map(lambda t: min(running_person_identifiers_df['Period'].unique(), key=lambda x: abs(x - t)),
                turning_points))

        # Add minimum and maximum period/frame of the interval we look at
        turning_points = sorted(set(turning_points) | set([min_period, max_period]))

        # Locate the x-mean values that belong to these points
        z = np.polyfit(x, y, 10)
        f = np.poly1d(z)
        turning_x = list(f(turning_points))

        # Find the relevant points by checking if it is the minimum and maximum period/frame of the interval or
        # if the points are sufficiently apart in both x and y coordinate
        points_x = []
        points_y = []
        for i, point in enumerate(turning_points):
            if point in [min_period, max_period]:
                points_x.append(point)
                points_y.append(turning_x[i])
            else:
                if abs(turning_x[turning_points.index(points_x[-1])] - turning_x[i]) > 200:
                    points_x.append(point)
                    points_y.append(turning_x[i])

        # Derive fragments
        fragments = [(i, j) for i, j in zip(points_x, points_x[1:]) if j - i > fps]

        # TODO: Odd place to do plotting, should remove
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
            mid = periods[len(periods) // 2]
            upper_bound = periods[(len(periods) // split) * (split - 1)]

            secondDerivative = fragment_series.diff().diff()  # calculating second derivative

            # print(fragment[0], lower_bound, mid, upper_bound, fragment[1])

            # plt.figure()
            # secondDerivative.plot()

            # At second derivative == 0, a person reaches his/her top speed
            # Around this point we set a certain confidence bound, for which we can be fairly sure that a person is running
            # Since the model tended to include slowing down a lot, a penalty is added relative to the start-up part
            # The map function makes sure that when no value falls within the interval that the upper or lower bound is assigned

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

        all_points = sorted(set(turning_frames) | set([min_period, max_period]))
        all_fragments = list(zip(all_points, all_points[1:]))

        # Splitting running fragments and turning fragments
        running_fragments = all_fragments[1::2]
        turning_fragments = all_fragments[::2]

        return running_fragments, turning_fragments, fragments
