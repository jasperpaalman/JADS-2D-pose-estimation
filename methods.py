import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import pandas as pd

from sklearn.cluster import DBSCAN

# pip install .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# pip install numpy --upgrade if numpy.multiarray error

from itertools import chain

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from math import sqrt

from data_extraction_methods import determine_rmse_threshold, rmse, amount_of_frames_to_look_back

connections = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]


def get_mean_x_per_person(person_period_division: Dict[int, Dict[int, any]]) \
        -> Dict[int, Dict[int, float]]:
    """
    Calculate the mean x-position of a person in a certain period

    :param person_period_division:
    :returns: a dictionary
    """
    return {person: {period: np.mean(coords[~(coords == 0).any(axis=1), 0])
                     for period, coords in time_coord_dict.items()}
            for person, time_coord_dict in person_period_division.items()}


def normalize_moved_distance_per_person(mean_x_per_person: Dict[int, Dict[int, any]]) -> Dict[int, int]:
    """
    Calculate moved distance by summing the absolute difference over periods
    Normalize moved distance per identified person over frames by including the average frame difference and the length
    of the number of frames included

    :param mean_x_per_person: A Persons containing their frames containing the mean x for that person for that frame.
    :return:
    """
    normalized_moved_distance_per_person = \
        {person: pd.Series(mean_x_dict).diff().abs().sum() / (
                np.diff(pd.Series(mean_x_dict).index).mean() * len(mean_x_dict))
         for person, mean_x_dict in mean_x_per_person.items()}

    return {key: value for key, value in normalized_moved_distance_per_person.items() if
            value == value}


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


def get_dbscan_subsets(maximum_normalized_distance: float, period_running_person_division_df: pd.DataFrame):
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

    return [list(i) for i in dbscan_subsets]


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


def determine_running_person_identifiers(
        period_running_person_division_df: pd.DataFrame,
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

    :param max_dist:
    :param max_rmse:
    :param max_dbscan_subset:
    :param period_running_person_division_df:


    :return running_person_identifiers:
    """

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
            iterative_main_traject_finder(period_running_person_division_df, running_person_identifiers, period, x, y,
                                          max_rmse, max_dist)

    for period in list(lower_periods)[::-1]:
        x, y, running_person_identifiers = \
            iterative_main_traject_finder(period_running_person_division_df, running_person_identifiers, period, x, y,
                                          max_rmse, max_dist)

    return running_person_identifiers


def get_running_and_turning_fragments(
        running_person_identifiers: List[int],
        mean_x_per_person: Dict,
        period_running_person_division_df,
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
        map(lambda t: min(running_person_identifiers_df['Period'].unique(), key=lambda x: abs(x - t)), turning_points))

    # Add minimum and maximum period/frame of the interval we look at
    turning_points = sorted(set(turning_points) | set([min_period, max_period]))

    # Locate the x-mean values that belong to these points
    z = np.polyfit(x, y, 10);
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
        z = np.polyfit(x, y, 10);
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


def get_clip_stats(clip_name):
    '''Get clip stats, when provided. Else return None.'''

    distances = ['16500']

    clip_name_split = clip_name.split('_')
    if clip_name_split[-1] in distances:
        length = int(clip_name_split[-3])
        weight = int(clip_name_split[-2])
        distance = int(clip_name_split[-1])
        return length, weight, distance
    else:
        return None


def euclidean_pairwise_distance(matrix):
    '''Given a matrix, calculates the pairwise distance between two rows. If the number of rows is not equal to 2 NaN is returned'''

    if matrix.shape[0] != 2:
        return np.nan

    return pairwise_distances(matrix[0].reshape(1, -1), matrix[1].reshape(1, -1))


def get_person_length_in_pixels(period_running_person_division, joint_confidence=0.5):
    '''Given the provided length of a person and some confidence bound on each joint ('gewricht' in Dutch) returns a measurement
    of a persons length in pixel values.'''

    # z value in the x,y,z coordinate output. Set a threshold to only include fairly certain coords

    # find all the coordinates of the person that are not empty and that exceed a set confidence level
    coord_list = [np.concatenate((np.arange(18).reshape(-1, 1), coords), axis=1)[(~(coords == 0).any(axis=1))
                                                                                 & (coords[:, 2] > joint_confidence)]
                  for period, person_dictionary in period_running_person_division.items()
                  for person, coords in person_dictionary.items()]

    # Check out: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose.png
    connections = [(0, 1), (1, 8), (1, 11), (8, 9), (11, 12), (9, 10),
                   (12, 13)]  # connections used for estimating length in pixels

    connection_lengths = []  # will contain averaged pixel length of the connections

    for connection in connections:
        connection_length = np.nanmean([euclidean_pairwise_distance(coords[np.isin(coords[:, 0], connection)][:, 1:3])
                                        for coords in coord_list])

        connection_lengths.append(connection_length)

    pixel_length = connection_lengths[0] + sum([np.mean([connection_lengths[i], connection_lengths[i + 1]])
                                                for i in range(len(connections))[1::2]])

    return pixel_length


def speed_via_length(period_running_person_division, running_fragments, length, fps, joint_confidence=0.5):
    '''Returns estimated speed in km/h per running fragment by using provided length as inference measurement.'''

    pixel_length = get_person_length_in_pixels(period_running_person_division, joint_confidence)
    length_in_meters = length / 100

    pixel_length_ratio = length_in_meters / pixel_length

    speed = []

    for start, end in running_fragments:
        start = int(round(start, 0))
        end = int(round(end, 0))

        start_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[start].values()])
        end_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[end].values()])

        x_diff = abs(end_x - start_x)

        meters_diff = pixel_length_ratio * x_diff

        fragment_speed = meters_diff / ((end - start) / fps) * 3.6

        speed.append(fragment_speed)

    return speed


def speed_via_distance(period_running_person_division, running_fragments, fragments, fps, distance=16500):
    '''Returns estimated speed in km/h per running fragment by using provided distance as inference measurement.'''

    distance_in_meters = distance / 1000

    bounds = []

    for start, end in fragments[1:3]:
        start_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[start].values()])
        end_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[end].values()])

        bounds = bounds + [start_x, end_x]

    lower_bound = min(bounds)
    upper_bound = max(bounds)

    pixel_distance = upper_bound - lower_bound

    pixel_distance_ratio = distance_in_meters / pixel_distance

    speed = []

    for start, end in running_fragments:
        start = min(period_running_person_division.keys(), key=lambda x: abs(x - start))
        end = min(period_running_person_division.keys(), key=lambda x: abs(x - end))

        start_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[start].values()])
        end_x = np.nanmean(
            [np.mean(coords[~(coords == 0).any(axis=1)][:, 0]) for coords in
             period_running_person_division[end].values()])

        x_diff = abs(end_x - start_x)

        meters_diff = pixel_distance_ratio * x_diff

        fragment_speed = meters_diff / ((end - start) / fps) * 3.6

        speed.append(fragment_speed)

    return speed


def plot_person(plottables, image_h, image_w, zoom=True, pad=3, sleep=0):
    """
    :param ax:
    :param f:
    :param plottables:
    """
    f, ax = plt.subplots(figsize=(14, 10))

    y_coords = [coords[~(coords == 0).any(axis=1)][:, 1]
                for period_dictionary in plottables.values() for coords in period_dictionary.values()]

    y_coords = list(chain.from_iterable(y_coords))

    cy = np.mean(y_coords)  # y center
    stdy = np.std(y_coords)  # y standard deviation

    ydiff = stdy * pad * 2  # total range of y

    aspect = image_w / image_h

    for t in sorted(plottables.keys()):

        for person in plottables[t].keys():
            plot_coords = plottables[t][person]

            coord_dict = {key: value for key, value in dict(enumerate(plot_coords[:, :2])).items() if 0 not in value}

            present_keypoints = set(coord_dict.keys())

            present_connections = [connection for connection in connections if
                                   len(present_keypoints & set(connection)) == 2]

            plot_lines = [np.transpose([coord_dict[a], coord_dict[b]]) for a, b in present_connections]

            plot_coords = plot_coords[~(plot_coords == 0).any(axis=1)]

            plt.scatter(x=plot_coords[:, 0], y=plot_coords[:, 1])

            for x, y in plot_lines:
                plt.plot(x, y)

            ax.annotate('Frame: {}'.format(t), xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(facecolor='red', alpha=0.5), fontsize=12)

            if zoom:
                ax.set_ylim(cy - stdy * pad, cy + stdy * pad)  # set y-limits by padding around the average center of y
                xlow, xhigh = ax.get_xlim()  # get x higher and lower limits
                xdiff = xhigh - xlow  # calculate the total range of x
                xpad = ((
                                ydiff * aspect) - xdiff) / 2  # calculate how much the xlimits should be padded on either side to set aspect ratio correctly
                ax.set_xlim(xlow - xpad, xhigh + xpad)  # set new limits
            else:
                ax.set_xlim([0, image_w])
                ax.set_ylim([-image_h, 0])

            f.canvas.draw()
            ax.clear()

            break

        time.sleep(sleep)


# Plotting coordinates of joints
def get_coord_list(period_person_division, running_person_identifiers, running_fragments):
    """"
    Returns a list of all coordinates

    :param period_person_division:
    :param running_person_identifiers:
    :param running_fragments:

    :return coord_list:
    """
    coord_list = []
    for n, running_fragment in enumerate(running_fragments):
        coord_list.append({})
        for period, period_dictionary in period_person_division.items():
            for person, coords in period_dictionary.items():
                if person in running_person_identifiers and running_fragment[0] <= period < running_fragment[1]:
                    coord_dict = {key: value for key, value in dict(enumerate(coords[:, :2])).items() if 0 not in value}
                    coord_list[n][period] = coord_dict
                    break
    return coord_list


def angle_between(p1, p2):
    """
    Calculate the clockwise angle between two points. Image drawing two lines from the origin (0,0) to both points and returning the
    angle between both in degrees.
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate(point, angle, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in degrees.
    """
    angle = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_rotation_angle(coord_df):
    """
    Given the coordinate dataframe with all the running coordinates, this function calculates what the degree to which the video is tilted (trend present).
    The rotation angle in degrees is returned to allow de-trending.
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
    if ynew[0] < ynew[-1]:
        ynew_origin = ynew - ynew.min()
        rotation_angle = -angle_between((xnew_origin[-1], ynew_origin[-1]), (xnew_origin[-1], 0))
    else:
        ynew_origin = ynew - ynew.max()
        rotation_angle = angle_between((xnew_origin[-1], 0), (xnew_origin[-1], ynew_origin[-1]))

    return rotation_angle


def reject_outliers(data, m=2):
    """
    Given an array of values return a boolean array indicating whether each entry is an outlier or not.
    """
    return abs(data - np.mean(data)) < m * np.std(data)


def process_coord_df(coord_df, period_running_person_division):
    """
    Process coord_df by de-trending and removing outliers and normalization.
    """

    # Get rotation angle for de-trending
    rotation_angle = get_rotation_angle(coord_df)
    # Remove trend by rotating x,y coordinate tuples
    coord_df["x"], coord_df["y"] = zip(
        *coord_df[['x', 'y']].apply(lambda d: rotate((d['x'], d['y']), rotation_angle), axis=1))
    # Remove outliers for each joint
    coord_df = coord_df[coord_df.groupby(['Point'])['y'].transform(reject_outliers).astype(bool)]

    pixel_length = np.mean(get_person_length_in_pixels(period_running_person_division))

    coord_df["x"] = coord_df["x"] / pixel_length
    coord_df["y"] = coord_df["y"] / pixel_length

    return coord_df


# To dataframe

def get_dataframe_from_coords(period_person_division, running_person_identifiers, running_fragments):
    """"
    Get a list of coordinates and turns this into a DataFrame to be used for analysis

    :param coord_list: List of List of dictionaries containing coordinates the run both ways.
    :return coord_df: A DataFrame containing all x and y coordinates of the runner during the run.

    The for loop when the 'Fragment' = i+1 is done should become a double for loop, also naming the video number, when adding more videos
    """
    coord_list = get_coord_list(period_person_division, running_person_identifiers, running_fragments)

    # More robust way of creating the coord_df
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

    period_running_person_division = {period: {person: coords for person, coords in period_dictionary.items()
                                               if person in running_person_identifiers}
                                      for period, period_dictionary in period_person_division.items()}
    period_running_person_division = dict(filter(lambda x: x[1] != {}, period_running_person_division.items()))

    coord_df = process_coord_df(coord_df, period_running_person_division)

    return coord_df


def forward_leaning_angle(coord_df):
    """
    Create forward leaning feature to be used in classification. The forward leaning feature describes to what extent a person
    leans forward. which could be an indicator of a good runner

    :param coord_df: A dataframe containing all relevant coördiantes observed in the video.
    :param running_fragments: Running intervals for a given video
    :return forward_leaning_per_fragment: Return a list with a forward leaning angle for each fragment
    """

    forward_leaning = []

    fragments = coord_df['Fragment'].unique()  # get all running fragments

    for fragment in fragments:
        fragment_df = coord_df[coord_df['Fragment'] == fragment]

        start = fragment_df[fragment_df['Frame'] == fragment_df['Frame'].min()]['x'].mean()  # start x
        end = fragment_df[fragment_df['Frame'] == fragment_df['Frame'].max()]['x'].mean()  # end x

        forward_leaning.append([])

        frames = fragment_df['Frame'].unique()  # unique frames for this fragment

        for frame in frames:
            df_sel = fragment_df[fragment_df['Frame'] == frame]
            forward_leaning_angles = []
            for points in [('Right Shoulder', 'Right Hip'), ('Left Shoulder', 'Left Hip')]:
                coords = df_sel[df_sel['Point'].isin(points)][['x', 'y']].as_matrix()
                if len(coords) == 2:
                    forward_leaning_point = coords[0] - coords[1]

                    # Determine direction
                    if end > start:  # direction is right
                        forward_leaning_angles.append(
                            angle_between((forward_leaning_point[0], forward_leaning_point[1]),
                                          (abs(forward_leaning_point[0]), 0)))
                    else:  # direction is left
                        forward_leaning_angles.append(angle_between((-abs(forward_leaning_point[0]), 0),
                                                                    (forward_leaning_point[0],
                                                                     forward_leaning_point[1])))
            if forward_leaning_angles != []:  # If points were found in this frame
                forward_leaning[fragment].append(np.mean(forward_leaning_angles))

    forward_leaning_per_fragment = [np.median(forward_leaning_list) for forward_leaning_list in forward_leaning]

    return forward_leaning_per_fragment


def to_feature_df(coord_df, video_number, period_running_person_division, running_fragments, fragments, fps):
    """
    Gets a DataFrame of coordinates and turns this into features.
    In this case, the standard deviation of movement vertically. Extension to also horizontally can be easily made in case this helps for discovering speed.

    :param coord_df: A dataframe containing all relevant coördiantes observed in the video.

    :return features_df: returns a dataframe containing standard deviations of all observed coordinates
    """

    # Set video number
    coord_df['video'] = video_number

    # extract basic std deviation features of all joints
    feature_df = coord_df.pivot_table(index=['video', 'Fragment'], columns='Point', values='y', aggfunc=np.std)

    # set video index
    feature_df['video'] = feature_df.index

    # Add value representing how much (in absoluut values) someone leaned forward
    feature_df['Forward_leaning'] = forward_leaning_angle(coord_df)

    feature_df['speed (km/h)'] = speed_via_distance(period_running_person_division, running_fragments, fragments, fps)

    return feature_df


def create_total_feature_df(coord_df, video_number, return_df, period_running_person_division, running_fragments,
                            fragments, fps):
    feature_df = to_feature_df(coord_df, video_number, period_running_person_division, running_fragments, fragments,
                               fps)
    if return_df is None:
        return_df = feature_df
    #         print(return_df)
    else:
        return_df = return_df.append(feature_df)
    return return_df


def get_plottables(period_person_division, running_person_identifiers, running_fragments, turning_fragments):
    """
    Function to construct all plottable files. In principle to be used for visualisation.
    """

    period_running_person_division = {period: {person: coords for person,
                                                                  coords in period_dictionary.items() if
                                               person in running_person_identifiers}
                                      for period, period_dictionary in period_person_division.items()}

    running_plottables = {
    period: {person: coords for person, coords in period_dictionary.items() if person in running_person_identifiers}
    for period, period_dictionary in period_person_division.items() if
    any(lower <= period <= upper for (lower, upper) in running_fragments)}

    turning_plottables = {
    period: {person: coords for person, coords in period_dictionary.items() if person in running_person_identifiers}
    for period, period_dictionary in period_person_division.items() if
    any(lower <= period <= upper for (lower, upper) in turning_fragments)}

    period_running_person_division = dict(filter(lambda x: x[1] != {}, period_running_person_division.items()))
    running_plottables = dict(filter(lambda x: x[1] != {}, running_plottables.items()))
    turning_plottables = dict(filter(lambda x: x[1] != {}, turning_plottables.items()))

    return period_running_person_division, running_plottables, turning_plottables
