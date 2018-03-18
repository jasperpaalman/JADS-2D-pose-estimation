import json
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
#import cv2
from sklearn.cluster import DBSCAN

# pip install .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# pip install numpy --upgrade if numpy.multiarray error

from itertools import chain
from itertools import groupby

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt


def get_openpose_output(location):
    """
    This function extracts all information generated by the openpose demo and converts it into a list of dictionaries.

    :param location: is the location of the .json files outputted by OpenPoseDemo.
    :return: List of List of dictionaries. So a list of frames, each frame consists of a list of dictionaries in which
    all identified people in the video are described using the coordinates of the observed joints.
    """
    people_per_file = []
    # coordinate files are ordered, so we can iterate through the folder in which the coordinates are stores for a clip
    # each file corresponds to a frame
    for path, subdirs, files in os.walk(location):
        for filename in files:
            coord_path = os.path.join(path, filename)
            with open(coord_path) as f:
                people_per_file.append(json.load(f)['people'])
    return people_per_file


def determine_video_meta_data(file_path):
    """"
    Extract the dimensions and frames per seconds from the video files

    :param file_path: Location of the video file
    :return: image_h image height, image_w image width, fps frames per second
    """
    cam = cv2.VideoCapture(file_path)
    ret_val, image = cam.read()
    image_h, image_w = image.shape[:2]  # getting clip resolution using opencv
    fps = cam.get(cv2.CAP_PROP_FPS)  # getting clip frames per second using opencv
    return image_h, image_w, fps


def rmse(a, b):
    """"
    Function which determines the rmse error between two sets of coordinates points

    :param a: coordinates a
    :param b: coordinates b
    :return: returns float
    """
    return sqrt(mean_squared_error(a, b))


def calc_center_coords_of_person(person_coords, used_joints):
    """"

    """
    cx = np.mean(person_coords[used_joints, 0])  # center x-coordinate
    cy = np.mean(person_coords[used_joints, 1])  # center y-coordinate
    return cx, cy


def determine_rmse_threshold(cx, cy, person_coords, used_joints):
    rmse_threshold = np.mean(pairwise_distances(np.array((cx, cy)).reshape(1, 2),
                                                person_coords[used_joints, :2]))
    return rmse_threshold


def join_lists_on_mutual_elements(plottable_subsets):
    all_moving_people = set(chain.from_iterable(plottable_subsets))
    for each in all_moving_people:
        components = [x for x in plottable_subsets if each in x]
        for i in components:
            plottable_subsets.remove(i)
        plottable_subsets += [list(set(chain.from_iterable(components)))]
    return plottable_subsets


def identify_people_over_multiple_frames(empty_joints, fps, period, period_person_division, person_coords, next_person):
    """

    :param next_person:
    :param empty_joints:
    :param fps:
    :param period:
    :param period_person_division:
    :param person_coords:

    :type period: int
    """
    # next_person = 0  # used to create a new person when the algorithm can't find a good person fit based on previous x frames
    best_person_fit = None  # Initially no best fit person in previous x frames is found
    if period == 0:  # period == 0 means no identified people exist, so we need to create them ourselves
        period_person_division[period][next_person] = person_coords  # create new next people since it is the first period
        next_person += 1
    else:
        min_rmse = 1000  # set sufficiently high rmse so it will be overwritten easily
        used_joints = list(set(range(18)) - empty_joints)  # only select used joints
        cx, cy = calc_center_coords_of_person(person_coords, used_joints)
        # set rmse_threshold equal to the mean distance of each used joint to the center
        rmse_threshold = determine_rmse_threshold(cx, cy, person_coords, used_joints)

        max_frame_diff = int(
            fps // 4)  # number of frames to look back, set to 0.25 sec rather than number of frames

        if period < max_frame_diff:
            j = period
        else:
            j = max_frame_diff

        for i in range(1, j + 1):  # for all possible previous periods within max_frame_diff
            for earlier_person in period_person_division[period - i].keys():  # compare with all people
                if earlier_person not in period_person_division[period].keys():  # if not already contained in current period
                    earlier_person_coords = period_person_division[period - i][earlier_person]
                    empty_joints_copy = empty_joints.copy()
                    empty_joints_copy = empty_joints_copy | set(np.where((earlier_person_coords == 0).all(axis=1))[0])
                    used_joints = list(set(range(18)) - empty_joints_copy)
                    if len(used_joints) == 0:
                        continue
                    # compute root mean squared error based only on mutual used joints
                    person_distance = rmse(earlier_person_coords[used_joints, :], person_coords[used_joints, :])
                    if person_distance < rmse_threshold:  # account for rmse threshold (only coordinates very close)
                        if person_distance < min_rmse:  # if best fit, when compared to previous instances
                            min_rmse = person_distance  # overwrite
                            best_person_fit = earlier_person  # overwrite
        if best_person_fit is not None:  # if a best person fit is found
            period_person_division[period][best_person_fit] = person_coords
        else:  # else create new next person
            period_person_division[period][next_person] = person_coords
            next_person += 1
    return period_person_division, next_person


# Getting plottable information per file
# Getting plottable information per file
def get_plottables_per_file_and_period_person_division(people_per_file, fps, connections):
    """"
    Troep code die we nog ff niet begrijpen maar het werkt haleluja

    """
    plottables_per_file = []  # used for plotting all the coordinates and connected body part lines

    # for each period/frame all 'people' are stored.
    # For a certain period this will allow us to look back in time at the previous x frames
    # in order to be able to group people in disjoint frames together

    period_person_division = {}  # Dict of dicts

    next_person = 0
    for period, file in enumerate(people_per_file):
        period_person_division[
            period] = {}  # for a frame (period) make a new dictionary in which to store the identified people

        plot_lines = []  # for plotting the entire video
        plot_coords = []  # for plotting the entire video
        plottables = {}  # new dictionary for a period, used for plotting entire video

        # coordinates of all people in this frame will be added to this list, to be iterated over later on
        # for plotting entire video
        coords = []

        for person in file:
            # append coords for this frame/file for each person in the right format
            coords.append(np.array([[x, -y, z] for x, y, z in np.reshape(person['pose_keypoints'], (18, 3))]))

            # information for identyfing people over disjoint frames
            person_coords = np.array([[x, -y, z] for x, y, z in np.reshape(person['pose_keypoints'], (18, 3))])
            # we don't want to base any computation on joints that are not present (==0), so we safe those indices that don't
            # contain any information
            empty_joints = set(np.where((person_coords == 0).all(axis=1))[0])

            # Identifying people over disjoint frames ###

            period_person_division, next_person = identify_people_over_multiple_frames(empty_joints, fps, period,
                                                                                       period_person_division, person_coords,
                                                                                       next_person)

        # For plotting the entire video ###

        for person_coords in coords:  # for all people in this frame
            plot_coords = plot_coords + list(
                person_coords[~(person_coords == 0).any(axis=1)])  # append present plottable coords

            # enumerate all x,y coordinate sets to be able to draw up lines
            # remove the ones that contain the value 0 --> joint not present
            coord_dict = {key: value for key, value in dict(enumerate(person_coords[:, :2])).items() if 0 not in value}

            present_keypoints = set(coord_dict.keys())  # only use joints that are present

            # get present connections: a connection contains 2 unique points, if a connection contains one of the keypoints that
            # is not present, the intersection of the connection with the present keypoints will be lower than 2
            # hence we end up with only the present connections
            present_connections = [connection for connection in connections if
                                   len(present_keypoints & set(connection)) == 2]

            # gather the connections, change the layout to fit matplotlib and extend the plot_lines list
            plot_lines = plot_lines + [np.transpose([coord_dict[a], coord_dict[b]]) for a, b in present_connections]

        if len(plot_coords) == 0:
            continue

        plot_coords = np.array(plot_coords)  # for easy indexing

        plottables['plot_coords'] = plot_coords
        plottables['plot_lines'] = plot_lines

        plottables_per_file.append(
            plottables)  # append plottables_per_file with the plottables dictionary for this frame
    return plottables_per_file, period_person_division


def plot_fit(plottables_per_file, period, f, ax, image_w, image_h):
    """"

    """
    plot_coords = plottables_per_file[period]['plot_coords']
    plot_lines = plottables_per_file[period]['plot_lines']
    plt.interactive(False)

    plt.scatter(x=plot_coords[:, 0], y=plot_coords[:, 1])

    for x, y in plot_lines:
        plt.plot(x, y)

    ax.set_xlim([0, image_w])
    ax.set_ylim([-image_h, 0])

    f.canvas.draw()
    ax.clear()


# Basically change the layout of the dictionary
# Now you first index based on the person and then you index based on the period
def get_person_period_division(period_person_division):
    person_period_division = {}
    for person in set(chain.from_iterable(period_person_division.values())):
        person_period_division[person] = {}
        for period in period_person_division.keys():
            period_dictionary = period_person_division[period]
            if person in period_dictionary:
                person_period_division[person][period] = period_dictionary[person]
    return person_period_division


# Calculate the mean x-position of a person in a certain period
def get_mean_x_per_person(person_period_division):
    return {person: {period: np.mean(coords[~(coords == 0).any(axis=1), 0])
                     for period, coords in time_coord_dict.items()}
            for person, time_coord_dict in person_period_division.items()}


# Calculate moved distance by summing the absolute difference over periods
# Normalize moved distance per identified person over frames by including the average frame difference and the length
# of the number of frames included
def normalize_moved_distance_per_person(mean_x_per_person):
    normalized_moved_distance_per_person = \
        {person: pd.Series(mean_x_dict).diff().abs().sum() / (np.diff(pd.Series(mean_x_dict).index).mean() * len(mean_x_dict))
         for person, mean_x_dict in mean_x_per_person.items()}

    return {key: value for key, value in normalized_moved_distance_per_person.items() if
            value == value}


# Finding person under observation based on clustering with DBSCAN

def get_person_plottables_df(mean_x_per_person, moving_people):
    return pd.DataFrame(
        [(period, person, x) for person, period_dict in mean_x_per_person.items() if person in moving_people
         for period, x in period_dict.items()], columns=['Period', 'Person', 'X mean'])


def get_dbscan_subsets(maximum_normalized_distance, person_plottables_df):
    db = DBSCAN(eps=maximum_normalized_distance, min_samples=1)

    db.fit(person_plottables_df[['Period', 'X mean']])

    person_plottables_df['labels'] = db.labels_

    maximum_label = person_plottables_df.groupby('labels').apply(len).sort_values(ascending=False).index[0]

    DBSCAN_subsets = person_plottables_df.groupby('labels')['Person'].unique().tolist()

    return [list(i) for i in DBSCAN_subsets]


def get_links(moving_people, mean_x_per_moving_person):
    links = []
    for n, person in enumerate(moving_people):
        x = mean_x_per_moving_person[person][:, 0]
        y = mean_x_per_moving_person[person][:, 1]

        # calculate polynomial
        z = np.polyfit(x, y, 1)
        f = np.poly1d(z)

        i = 2  # how many periods back and forth to compare

        if n < i:
            i = n

        for j in range(1, i + 1):
            if n > 0:
                previous_person = moving_people[n - j]

                previous_x = mean_x_per_moving_person[previous_person][:, 0]
                previous_y = mean_x_per_moving_person[previous_person][:, 1]

                previous_rmse = rmse(f(previous_x), previous_y)

                links.append((previous_person, person, previous_rmse))

            if n < len(moving_people) - j:
                next_person = moving_people[n + j]

                next_x = mean_x_per_moving_person[next_person][:, 0]
                next_y = mean_x_per_moving_person[next_person][:, 1]

                next_rmse = rmse(f(next_x), next_y)

                links.append((person, next_person, next_rmse))
    return links


# Averaging RMSE between links
def get_linked_people(maximum_normalized_distance, links):
    link_rmse = np.array(
        [(key, np.mean(np.array(list(group))[:, 2])) for key, group in groupby(links, lambda x: (x[0], x[1]))])
    # Use threshold on RMSE to get linked people
    linked_people = link_rmse[link_rmse[:, 1] < maximum_normalized_distance * 2][:, 0]

    # Setting in right format
    return [list(i) for i in linked_people]


def iterative_main_traject_finder(person_plottables_df, plottable_people, period, x, y, max_rmse):
    """Given a period that needs to be tested and some x,y coordinate set to extrapolate from, this function tests,
    based on the maximum RMSE, if the point(s) within this period are comparable with the current region.
    The x,y coordinates are returned as well as the updated plottable people set.



    """

    best_point = None

    z = np.polyfit(x, y, 10)  # fit polynomial with sufficient degree to the datapoints
    f = np.poly1d(z)

    # retrieve values that belong to this period (can contain more than one point, when noise is present)
    period_selection = person_plottables_df[person_plottables_df['Period'] == period][['Period', 'Person', 'X mean']].values

    # for each of these points check the RMSE
    for period, person, x_mean in period_selection:
        rmse_period = rmse([f(period)], [x_mean])
        if rmse_period < max_rmse:
            max_rmse = rmse_period
            best_point = (period, x_mean, person)

    if best_point != None:
        x.append(best_point[0])
        y.append(best_point[1])
        plottable_people = plottable_people | set([int(best_point[2])])

    return x, y, plottable_people


def determine_plottable_people(person_plottables_df, max_dbscan_subset, max_rmse):
    """This function takes the largest DBSCAN subset as a starting point and starts expanding to periods that are not
    yet covered. For each period not covered yet, the points that are already included are used to create a polynomial
    function to extrapolate from. The points contained within the period are compared and one or zero points can be chosen
    to be included in the main traject/region, which depends on the maximum RMSE that is set. If rmse of no point for a period
    lies below the maximum RMSE, no point is included and we move over to the next period in line. The periods lower than
    the initially covered region by DBSCAN is indicated as the lower_periods, the periods higher as the upper_periods."""

    plottable_people = set(max_dbscan_subset)  # set-up plottable people set

    # Make a selection of the dataframe that is contained within the current initial region
    df_sel = person_plottables_df[person_plottables_df['Person'].isin(max_dbscan_subset)].sort_values('Period')

    x = df_sel['Period'].tolist()  # starting x list
    y = df_sel['X mean'].tolist()  # starting y list

    # Region lower and upper bound
    region_lower_bound = person_plottables_df[person_plottables_df['Person'] == min(max_dbscan_subset)]['Period'].min()
    region_upper_bound = person_plottables_df[person_plottables_df['Person'] == max(max_dbscan_subset)]['Period'].max()

    # Determine lower and upper periods to cover
    lower_periods = set(range(person_plottables_df['Period'].min(), region_lower_bound)) & set(person_plottables_df['Period'])
    upper_periods = set(range(region_upper_bound + 1, person_plottables_df['Period'].max())) & set(person_plottables_df['Period'])

    for period in upper_periods:
        x, y, plottable_people = \
            iterative_main_traject_finder(person_plottables_df, plottable_people, period, x, y, max_rmse)

    for period in list(lower_periods)[::-1]:
        x, y, plottable_people = \
            iterative_main_traject_finder(person_plottables_df, plottable_people, period, x, y, max_rmse)

    return plottable_people


def get_running_fragments(plottable_people, mean_x_per_person, person_plottables_df):
    """
    Given the identified plottable people (running person/person under observation),
    this function returns the divided running fragments. That is, each running fragment is a
    person running from one side to the other.

    :param plottable_people:
    :param mean_x_per_person:
    :param person_plottables_df:

    :return :
    """
    # Retrieve dataframe, but only select plottable people
    plottable_people_df = person_plottables_df[person_plottables_df['Person'].isin(plottable_people)].sort_values('Period')

    x = plottable_people_df['Period'].values
    y = plottable_people_df['X mean'].values

    min_period = plottable_people_df['Period'].min()
    max_period = plottable_people_df['Period'].max()

    # fit polynomial with sufficient degree to the datapoints
    z = np.polyfit(x, y, 10)
    f = np.poly1d(z)

    # Construct new smooth line using the polynomial function
    xnew = np.linspace(min_period, max_period, num=len(x) * 10, endpoint=True)
    ynew = f(xnew)

    # Determine optima indexes for xnew and ynew
    optima_ix = np.diff(np.sign(np.diff(ynew))).nonzero()[0] + 1  # local min+max

    # The optima reflect the turning points, which can be retrieved in terms of frames
    turning_points = xnew[optima_ix].astype(int)

    plt.plot(xnew, ynew)
    plt.plot(xnew[optima_ix], ynew[optima_ix], "o", label="min")
    plt.title('Finding turning points (optima) and deriving running fragments')
    plt.xlabel('Frames')
    plt.ylabel('X position')

    pd.DataFrame({key: value for key, value in mean_x_per_person.items() if key in plottable_people}).plot()
    plt.title('All coordinates that will be used in further analyses')
    plt.xlabel('Frames')
    plt.ylabel('X position')

    # Add minimum and maximum period/frame of the interval we look at
    turning_points = sorted(set(turning_points) | set([min_period, max_period]))

    # Derive running fragments by only taking fragments for which the start and end frame are a minimum of 10 frames apart
    running_fragments = [(i, j) for i, j in zip(turning_points, turning_points[1:]) if j - i > 10]

    # We should somehow include a general cut-off around the turning points, to remove the noise of a person turning

    return running_fragments


def plot_person(plottables, f, ax, connections):
    for person in plottables.keys():
        plot_coords = plottables[person]

        coord_dict = {key: value for key, value in dict(enumerate(plot_coords[:, :2])).items() if 0 not in value}

        present_keypoints = set(coord_dict.keys())

        present_connections = [connection for connection in connections if
                               len(present_keypoints & set(connection)) == 2]

        plot_lines = [np.transpose([coord_dict[a], coord_dict[b]]) for a, b in present_connections]

        plot_coords = plot_coords[~(plot_coords == 0).any(axis=1)]

        plt.scatter(x=plot_coords[:, 0], y=plot_coords[:, 1])

        for x, y in plot_lines:
            plt.plot(x, y)

    ax.set_xlim([plot_coords[:, 0].min() - 100, plot_coords[:, 0].max() + 100])
    ax.set_ylim([plot_coords[:, 1].min() - 100, plot_coords[:, 1].max() + 100])

    #     ax.set_xlim([0, image_w])
    #     ax.set_ylim([-image_h, 0])

    f.canvas.draw()
    ax.clear()


# Plotting coordinates of joints
def prepare_data_for_plotting(period_person_division, plottable_people, running_fragments):
    """"

    :param period_person_division:
    :param plottable_people:
    :param running_fragments:

    :return coord_list:
    """
    coord_list = []
    for n, running_fragment in enumerate(running_fragments):
        coord_list.append([])
        for period, period_dictionary in period_person_division.items():
            for person, coords in period_dictionary.items():
                if person in plottable_people and running_fragment[0] <= period < running_fragment[1]:
                    coord_dict = {key: value for key, value in dict(enumerate(coords[:, :2])).items() if 0 not in value}
                    coord_list[n].append(coord_dict)
                    break

    return coord_list


# To dataframe

def get_dataframe_from_coords(coord_list):
    """"
    Get a list of coordinates and turns this into a DataFrame to be used for analysis
    :param coord_list: List of List of dictionaries containing coordinates the run both ways.
    :return coord_df: A DataFrame containing all x and y coordinates of the runner during the run.
    
    The for loop when the 'Fragment' = i+1 is done should become a double for loop, also naming the video number, when adding more videos
    """
    coord_df = pd.DataFrame(coord_list[0]).append(pd.DataFrame(coord_list[1]), ignore_index=True)
    for i in range(len(coord_list)):
        if i == 0:
            coord_df = pd.DataFrame(coord_list[i])
            coord_df['Fragment'] = i +1
        else:
            temp_df = pd.DataFrame(coord_list[i])
            temp_df['Fragment'] = i + 1
            coord_df = coord_df.append(temp_df)

    coord_df.columns = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Hand',
                        'Left Shoulder', 'Left Elbow', 'Left Hand', 'Right Hip',
                        'Right Knee', 'Right Foot', 'Left Hip', 'Left Knee',
                        'Left Foot', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Fragment']

    # add the frame number
    coord_df['Frame'] = coord_df.index

    # melt the dataframe to get the locations in one row
    coord_df = pd.melt(coord_df, id_vars=['Frame', 'Fragment'], var_name='Point', value_name='Location')

    # remove unecessary signs, for some unclear reason this is not necessary after splitting
    # coord_df['Location'] = coord_df.Location.apply(lambda x: str(x).replace('[', ''))
    # coord_df['Location'] = coord_df.Location.apply(lambda x: str(x).replace(']', ''))

    # split up the coordinates and put them into separate columns
    coord_df['Split'] = coord_df.Location.apply(lambda x: str(x).split('  '))
    coord_df['x'] = coord_df.Location.str.get(0)
    coord_df['y'] = coord_df.Location.str.get(1)

    # delete irrelevant columns
    del coord_df['Split']
    del coord_df['Location']

    return coord_df


def to_feature_df(coord_df):
    """
    Gets a DataFrame of coordinates and turns this into features.
    In this case, the standard deviation of movement vertically. Extension to also horizontally can be easily made in case this helps for discovering speed.

    :param coord_df: A dataframe containing all relevant coördiantes observed in the video.

    :return features_df: returns a dataframe containing standard deviations of all observed coordinates
    """
    coord_df['video'] = 1  # needs to be used as itterator in later version for multiple video's

    y_df = coord_df.pivot_table(index=['video', 'Fragment'], columns='Point', values='y', aggfunc=np.std)
    #y_df.columns = [str(col) + '_y' for col in y_df.columns]
    y_df['video'] = y_df.index
    feature_df = y_df

    return feature_df, coord_df
    # return y_df

def forward_leaning(feature_df, coord_df):
    feature_df['Forward_leaning'] = 0
    fragments = set(coord_df.Fragment)

    for i in range(len(fragments)):
        fragment_df = coord_df[coord_df['Fragment'] == i+1]
        shoulder_df = fragment_df[fragment_df['Point'] == 'Right Shoulder']
        hip_df = fragment_df[fragment_df['Point'] == 'Right Hip']
        frames = set(fragment_df.Frame)
        temp_sum = 0
        frame_count = 0
        for j in range(len(frames)):
            difference = shoulder_df.iloc[j, 3] - hip_df.iloc[j, 3]
            #couldn't think of a smarter way to not take the nan values into account for the average
            if difference > 1:
                frame_count += 1
                temp_sum += difference
            if difference < -1:
                frame_count += 1
                temp_sum += difference
            #print(difference)
        #print(temp_sum)
        feature_df.iloc[i, 19] = abs(temp_sum / frame_count)  
    return feature_df

# Plotly functions to make coördinates more insightfull
# todo: @collin kan jij nog naar deze functie kijken
def plotly_scatterplot(pointlist):
    points = []
    for i in range(len(pointlist)):
        pointdf = coord_df[(coord_df['Point'] == pointlist[i])]
        trace = go.Scatter(
            # df = worldbank[(worldbank['Country'] == 'Belgium')],
            x=pointdf['x'],
            y=pointdf['y'],
            mode='markers',
            name=pointlist[i],
            text=pointdf['Frame'],
            opacity=0.7,
            marker=dict(
                size='5',  # makes the dots invisible, can't get rid of them somehow
                color=i
            )
        )
        points.append(trace)

    layout = dict(
        title='Open Pose coordinate tracker',
        hovermode='closest',
        yaxis=dict(
            #         rangeslider=dict(),
            #         type='date'
            # range=[-600, -300]
        )
        #     ylim = (-600, 300)
    )

    fig = dict(data=points, layout=layout)

    return fig


def plotly_boxplot(pointlist, coord_df):
    # TODO: @collin Why do we do almost the same thing twice?
    # we don't, it is just the syntax of plotly
    points = []
    for i in range(len(pointlist)):
        pointdf = coord_df[(coord_df['Point'] == pointlist[i])]
        trace = go.Box(
            # df = worldbank[(worldbank['Country'] == 'Belgium')],
            y=pointdf['y'],
            # boxpoints = 'all',
            name=pointlist[i],
            #         text = pointdf['Frame'],
            opacity=0.7,
            #         marker=dict(
            #             size='5', #makes the dots invisible, can't get rid of them somehow
            #             color = i
            #         )
        )
        points.append(trace)

    layout = dict(
        title='Open Pose',
        hovermode='closest',
        yaxis=dict(
            #         rangeslider=dict(),
            #         type='date'
            # range=[-600, -300]
        )
        #     ylim = (-600, 300)
    )

    fig = dict(data=points, layout=layout)

    return fig
