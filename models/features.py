from pandas import DataFrame
import numpy as np
import math
import pandas as pd
import warnings

from models.preprocessor import Preprocessor
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings('ignore')

class Features:

    def __init__(self, feature_df: DataFrame):
        self.feature_df = feature_df

    def to_json(self, path_or_buff):
        self.feature_df.to_json(path_or_buff)

    def from_json(self, path_or_buff):
        self.feature_df = pd.read_json(path_or_buff)

    @staticmethod
    def from_preprocessor(preprocessor: Preprocessor):
        return Features(Features.__get_features(preprocessor))

    @staticmethod
    def __get_features(preprocessor: Preprocessor) -> DataFrame:
        """
        Given instance of Preprocessor class returns a Dataframe containing all features for this video

        :param preprocessor: Instance of Preprocessor class

        :return: Dataframe containing all features for this video
        """
        coord_df = Features.get_dataframe_from_coords(
            preprocessor.get_period_person_division(),
            preprocessor.get_running_person_identifiers(),
            preprocessor.get_running_fragments())

        period_running_person_division, running_plottables, turning_plottables = Features.get_plottables(
            preprocessor.get_period_person_division(),
            preprocessor.get_running_person_identifiers(),
            preprocessor.get_running_fragments(),
            preprocessor.get_turning_fragments()
        )

        feature_df = Features.to_feature_df(
            coord_df,
            preprocessor.source,
            period_running_person_division,
            preprocessor.get_running_fragments(),
            preprocessor.get_fragments(),
            preprocessor.frame_rate)

        print('processed video: ', preprocessor.source)

        return feature_df

    @staticmethod
    def get_coord_list(period_person_division, running_person_identifiers, running_fragments):
        """"
        Returns a list of all coordinates for running fragments

        :param period_person_division: Data strucure containing per frame all persons and their corresponding
                                       coordinates
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param running_fragments: The estimated fragments where the person under observation is running

        :return coord_list: A list with for each running fragment a dictionary with frame as key and a coordinate
                            dictionary as value
        """
        coord_list = []
        for n, running_fragment in enumerate(running_fragments):
            coord_list.append({})  # Instantiate dictionary
            for period, person_dictionary in period_person_division.items():
                for person, coords in person_dictionary.items():
                    if person in running_person_identifiers and running_fragment[0] <= period < running_fragment[1]:
                        # Enumerate all (x,y) combinations following the indices of joints from Openpose and create dict
                        coord_dict = {key: value for key, value in dict(enumerate(coords[:, :2])).items() if
                                      0 not in value}
                        coord_list[n][period] = coord_dict  # Add coord_dict to the right fragment and the right period
                        break  # break to only keep one person per frame
        return coord_list

    @staticmethod
    def angle_between(p1, p2):
        """
        Calculates the clockwise angle between two points. Imagine drawing two lines from the origin (0,0) to both
        points and returning the angle between both in degrees.

        :param p1: Point 1
        :param p2: Point 2

        :return: Angle between the two points in degrees
        """
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * math.pi))

    @staticmethod
    def rotate(point, angle, origin=(0, 0)):
        """
        Rotate a point (x,y) counterclockwise by a given angle around a given origin. The angle should be given in
        degrees.

        :param point: Point as (x,y)
        :param angle: Rotation angle in degrees
        :param origin: Origin to rotate around

        :return: Rotated point as (x,y)
        """
        angle = math.radians(angle)

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    @staticmethod
    def get_rotation_angle(coord_df):
        """
        Given the coordinate dataframe with all the running coordinates, this function calculates the degree to
        which the video is tilted (trend present). The rotation angle in degrees is returned to allow de-trending.

        :param coord_df: A dataframe containing all relevant coördiantes observed in the video.

        :return: Rotation angle in degrees by which all coordinates should be rotated to detrend the video
        """

        running_coords = coord_df[['x', 'y']].as_matrix()

        x_coords = running_coords[:, 0]
        y_coords = running_coords[:, 1]

        # Fit linear line
        z = np.polyfit(x_coords, y_coords, 1)
        f = np.poly1d(z)

        # Construct new smooth line using the polynomial function
        xnew = np.linspace(x_coords.min(), x_coords.max(), num=len(x_coords) * 10, endpoint=True)
        ynew = f(xnew)

        # Move line over to start at (0,0) and get the rotation_angle
        xnew_origin = xnew - x_coords.min()
        if ynew[0] < ynew[-1]:  # determine the direction of the line
            ynew_origin = ynew - ynew.min()
            rotation_angle = - Features.angle_between((xnew_origin[-1], ynew_origin[-1]), (xnew_origin[-1], 0))
        else:
            ynew_origin = ynew - ynew.max()
            rotation_angle = Features.angle_between((xnew_origin[-1], 0), (xnew_origin[-1], ynew_origin[-1]))

        return rotation_angle

    @staticmethod
    def reject_outliers(data, m=2):
        """
        Given an array of values return a boolean array indicating whether each entry is an outlier or not.

        :param data: Array of data points
        :param m: Float indicating the number of standard deviations to set as threshold

        :return: Returns a boolean array indicating if a point is not an outlier
        """
        return abs(data - np.mean(data)) < m * np.std(data)

    @staticmethod
    def process_coord_df(coord_df, period_running_person_division):
        """
        Process coord_df by de-trending and removing outliers and normalization.

        :param coord_df: A dataframe containing all relevant coördinates observed in the video
        :param period_running_person_division: Data strucure containing per frame all running persons and their
                                            corresponding coordinates

        :return: A dataframe containing all relevant procesed coordinates observed in the video
        """

        # Get rotation angle for de-trending
        rotation_angle = Features.get_rotation_angle(coord_df)
        # Remove trend by rotating x,y coordinate tuples
        coord_df["x"], coord_df["y"] = zip(
            *coord_df[['x', 'y']].apply(lambda d: Features.rotate((d['x'], d['y']), rotation_angle), axis=1))
        # Remove outliers for each joint
        coord_df = coord_df[coord_df.groupby(['Point'])['y'].transform(Features.reject_outliers).astype(bool)]
        # Get length of person in pixels to normalize coordinate values by
        pixel_length = np.mean(Features.get_person_length_in_pixels(period_running_person_division))

        coord_df["x"] = coord_df["x"] / pixel_length
        coord_df["y"] = coord_df["y"] / pixel_length

        return coord_df

    @staticmethod
    def get_dataframe_from_coords(period_person_division, running_person_identifiers, running_fragments):
        """
        Get a list of coordinates and turns this into a DataFrame to be used for analysis

        :param period_person_division: Data strucure containing per frame all persons and their corresponding
                                       coordinates
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param running_fragments: The estimated fragments where the person under observation is running

        :return: A dataframe containing all relevant coördiantes observed in the video.
        """
        coord_list = Features.get_coord_list(period_person_division, running_person_identifiers, running_fragments)

        # Create coord_df
        coord_df = pd.DataFrame(
            [(n, frame, ix, *coords) for n, period_dict in enumerate(coord_list) for frame, coord_dict in
             period_dict.items()
             for ix, coords in coord_dict.items()], columns=['Fragment', 'Frame', 'Point', 'x', 'y'])

        # Numeric to name dictionary
        replace_dict = dict(enumerate(['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Hand',
                                       'Left Shoulder', 'Left Elbow', 'Left Hand', 'Right Hip',
                                       'Right Knee', 'Right Foot', 'Left Hip', 'Left Knee',
                                       'Left Foot', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']))

        # Turn numerics into names
        coord_df['Point'] = coord_df['Point'].replace(replace_dict)

        # Get period_running_person_division for processing coord_df
        period_running_person_division = {period: {person: coords for person, coords in period_dictionary.items()
                                                   if person in running_person_identifiers}
                                          for period, period_dictionary in period_person_division.items()}
        period_running_person_division = dict(filter(lambda x: x[1] != {}, period_running_person_division.items()))

        coord_df = Features.process_coord_df(coord_df, period_running_person_division)

        return coord_df

    @staticmethod
    def forward_leaning_angle(coord_df):
        """
        Create forward leaning feature to be used in classification. The forward leaning feature describes to what
        extent a person leans forward, which could be an indicator of a good runner

        :param coord_df: A dataframe containing all relevant coördiantes observed in the video.

        :return forward_leaning_per_fragment: Return a list with a forward leaning angle for each running fragment
        """

        forward_leaning = []

        fragments = coord_df['Fragment'].unique()  # get all running fragments

        for fragment in fragments:  # for each running fragment
            fragment_df = coord_df[coord_df['Fragment'] == fragment]  # make data selection

            start = fragment_df[fragment_df['Frame'] == fragment_df['Frame'].min()]['x'].mean()  # start x
            end = fragment_df[fragment_df['Frame'] == fragment_df['Frame'].max()]['x'].mean()  # end x

            forward_leaning.append([])  # instantiate list for this fragment

            frames = fragment_df['Frame'].unique()  # unique frames for this fragment

            for frame in frames:
                df_sel = fragment_df[fragment_df['Frame'] == frame]
                forward_leaning_angles = []
                # For the right side as well as the left side derive an angle
                for points in [('Right Shoulder', 'Right Hip'), ('Left Shoulder', 'Left Hip')]:
                    # Get x and y coordinates of shoulder and hip points
                    coords = df_sel[df_sel['Point'].isin(points)][['x', 'y']].as_matrix()
                    if len(coords) == 2:  # Only if both the hip and shoulder are present
                        forward_leaning_point = coords[0] - coords[1]  # shift point to the origin (0,0) by subtraction

                        # Determine direction
                        if end > start:  # direction is right
                            forward_leaning_angles.append(
                                Features.angle_between((forward_leaning_point[0], forward_leaning_point[1]),
                                                       (abs(forward_leaning_point[0]), 0)))
                        else:  # direction is left
                            forward_leaning_angles.append(Features.angle_between((-abs(forward_leaning_point[0]), 0),
                                                                                 (forward_leaning_point[0],
                                                                                  forward_leaning_point[1])))
                if len(forward_leaning_angles) > 0:  # If points were found in this frame
                    # Take mean over left and right forward leaning estimations
                    forward_leaning[fragment].append(np.mean(forward_leaning_angles))

        # For each fragment (multiple frames per fragment) take the median forward leaning angle to limit the
        # effect of outliers
        forward_leaning_per_fragment = [np.median(forward_leaning_list) for forward_leaning_list in forward_leaning]

        return forward_leaning_per_fragment

    @staticmethod
    def to_feature_df(coord_df, source, period_running_person_division, running_fragments, fragments, fps):
        """
        Gets a DataFrame of coordinates and turns this into features.
        In this case, the standard deviation of movement vertically and the forward leaning angle

        :param coord_df: A dataframe containing all relevant coördiantes observed in the video.
        :param source: Name of the video
        :param period_running_person_division: Data strucure containing per frame all running persons and their
                                            corresponding coordinates
        :param running_fragments: The estimated fragments where the person under observation is running
        :param fragments: The estimated fragments containing start-up, run and slow-down fases
        :param fps: Frames per second of the shot video

        :return: A Dataframe with features to apply analysis to
        """

        # Set video number
        coord_df['source'] = source

        # extract basic std deviation features of all joints
        feature_df = coord_df.pivot_table(index=['source', 'Fragment'], columns='Point', values='y', aggfunc=np.std)

        # # set video index
        feature_df['source'] = source

        # Add value representing how much (in absoluut values) someone leaned forward
        feature_df['Forward_leaning'] = Features.forward_leaning_angle(coord_df)

        feature_df['speed (km/h)'] = Features.speed_via_distance(period_running_person_division, running_fragments,
                                                                 fragments,
                                                                 fps)

        return feature_df

    @staticmethod
    def euclidean_pairwise_distance(matrix):
        """
        Given a matrix, calculates the pairwise distance between two rows. If the number of rows is not equal to 2
        NaN is returned

        :param matrix: A matrix with data points

        :return: Returns euclidean pairwise distance if matrix contains two rows, else NaN
        """

        if matrix.shape[0] != 2:
            return np.nan

        return pairwise_distances(matrix[0].reshape(1, -1), matrix[1].reshape(1, -1))

    @staticmethod
    def get_person_length_in_pixels(period_running_person_division, joint_confidence=0.5):
        """
        Given the provided length of a person and some confidence bound on each joint ('gewricht' in Dutch) returns a
        measurement of a persons length in pixel values.

        :param period_running_person_division: Data strucure containing per frame all running persons and their
                                            corresponding coordinates
        :param joint_confidence: Confidence level between 0 and 1, which limits the coordinates that are used in
                                 calculations

        :return: Returns the length of the person under observation in pixels
        """

        # find all the coordinates of the person that are not empty and that exceed a set confidence level
        coord_list = [np.concatenate((np.arange(18).reshape(-1, 1), coords), axis=1)[(~(coords == 0).any(axis=1))
                                                                                     & (coords[:,
                                                                                        2] > joint_confidence)]
                      for period, person_dictionary in period_running_person_division.items()
                      for person, coords in person_dictionary.items()]

        # Check out: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose.png
        connections = [(0, 1), (1, 8), (1, 11), (8, 9), (11, 12), (9, 10),
                       (12, 13)]  # connections used for estimating length in pixels

        connection_lengths = []  # will contain averaged pixel length of the connections

        for connection in connections:
            connection_length = np.nanmean(
                [Features.euclidean_pairwise_distance(coords[np.isin(coords[:, 0], connection)][:, 1:3])
                 for coords in coord_list])

            connection_lengths.append(connection_length)

        pixel_length = connection_lengths[0] + sum([np.mean([connection_lengths[i], connection_lengths[i + 1]])
                    for i in range(len(connections))[1::2]])  # mean used to average right and left estimations

        return pixel_length

    @staticmethod
    def speed_via_distance(period_running_person_division, running_fragments, fragments, fps, distance=16500):
        """
        Returns estimated speed in km/h per running fragment by using provided distance as inference measurement.

        :param period_running_person_division: Data strucure containing per frame all running persons and their
                                            corresponding coordinates
        :param running_fragments: The estimated fragments where the person under observation is running
        :param fragments: The estimated fragments containing start-up, run and slow-down fases
        :param fps: Frames per second of the shot video
        :param distance: Distance in mm that is travelled each fragment

        :return: An array with for each running fragment an estimated speed in km/h
        """

        distance_in_meters = distance / 1000

        # get lower_bound turning point using second fragment
        lower_bound = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[fragments[1][0]].values()])

        # get upper_bound turning point using second fragment
        upper_bound = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[fragments[1][1]].values()])

        pixel_distance = upper_bound - lower_bound

        pixel_distance_ratio = distance_in_meters / pixel_distance

        speed = []

        for start, end in running_fragments:
            start = min(period_running_person_division.keys(), key=lambda x: abs(x - start))  # find start frame
            end = min(period_running_person_division.keys(), key=lambda x: abs(x - end))  # find end frame

            start_x = np.nanmean(
                [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
                 period_running_person_division[start].values()])  # find start mean x
            end_x = np.nanmean(
                [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
                 period_running_person_division[end].values()])  # find end mean x

            x_diff = abs(end_x - start_x)

            meters_diff = pixel_distance_ratio * x_diff  # to meters

            fragment_speed = meters_diff / ((end - start) / fps) * 3.6

            speed.append(fragment_speed)

        return speed

    @staticmethod
    def speed_via_length(period_running_person_division, running_fragments, length, fps, joint_confidence=0.5):
        """
        Returns estimated speed in km/h per running fragment by using provided length as inference measurement.

        :param period_running_person_division: Data strucure containing per frame all running persons and their
                                            corresponding coordinates
        :param running_fragments: The estimated fragments where the person under observation is running
        :param length: Length of person in cm
        :param fps: Frames per second of the shot video
        :param joint_confidence: Confidence level between 0 and 1, which limits the coordinates that are used in
                                 calculations

        :return: An array with for each running fragment an estimated speed in km/h
        """

        pixel_length = Features.get_person_length_in_pixels(period_running_person_division, joint_confidence)
        length_in_meters = length / 100

        pixel_length_ratio = length_in_meters / pixel_length

        speed = []

        for start, end in running_fragments:
            start = min(period_running_person_division.keys(), key=lambda x: abs(x - start))  # find start frame
            end = min(period_running_person_division.keys(), key=lambda x: abs(x - end))  # find end frame

            start_x = np.nanmean(
                [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
                 period_running_person_division[start].values()])  # find start mean x
            end_x = np.nanmean(
                [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
                 period_running_person_division[end].values()])  # find end mean x

            x_diff = abs(end_x - start_x)

            meters_diff = pixel_length_ratio * x_diff

            fragment_speed = meters_diff / ((end - start) / fps) * 3.6

            speed.append(fragment_speed)

        return speed

    @staticmethod
    def get_plottables(period_person_division, running_person_identifiers, running_fragments, turning_fragments):
        """
        Function to construct all plottable files. In principle to be used for visualisation.

        :param period_person_division: Data strucure containing per frame all persons and their corresponding
                                       coordinates
        :param running_person_identifiers: The indices of identified 'people' that belong to the running person
        :param running_fragments: The estimated fragments where the person under observation is running
        :param turning_fragments: The estimated fragments where the person under observation is turning

        :return: Plottable information for each distinct part in a fragment. The data structury is similar
                 to the person_period_division since it gives a data structure containing per person all frames
                 and the coordinates of that person in that frame.
        """

        period_running_person_division = {period: {person: coords
                                                   for person, coords in period_dictionary.items() if
                                                   person in running_person_identifiers}
                                          for period, period_dictionary in period_person_division.items()}

        running_plottables = {
            period: {person: coords for person, coords in period_dictionary.items() if
                     person in running_person_identifiers}
            for period, period_dictionary in period_person_division.items() if
            any(lower <= period <= upper for (lower, upper) in running_fragments)}

        turning_plottables = {
            period: {person: coords for person, coords in period_dictionary.items() if
                     person in running_person_identifiers}
            for period, period_dictionary in period_person_division.items() if
            any(lower <= period <= upper for (lower, upper) in turning_fragments)}

        period_running_person_division = dict(filter(lambda x: x[1] != {}, period_running_person_division.items()))
        running_plottables = dict(filter(lambda x: x[1] != {}, running_plottables.items()))
        turning_plottables = dict(filter(lambda x: x[1] != {}, turning_plottables.items()))

        return period_running_person_division, running_plottables, turning_plottables
