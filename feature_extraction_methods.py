from typing import List

import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def create_total_feature_df(coord_df, video_number, return_df):
    """
    Function that combines the different feature data frames from the different video's into one big frame
    :param coord_df: A dataframe containing all relevant coördiantes observed in the video.
    :param video_number: The index of the video currently being analyzed
    :param return_df: DataFrame containing the combined features of all video's
    :return return_df: DataFrame containing the combined features of all video's
    """
    feature_df = to_feature_df(coord_df, video_number)
    if return_df is None:
        return_df = feature_df
        print(return_df)
    else:
        return_df = return_df.append(feature_df)
    return return_df


def to_feature_df(coord_df: pd.DataFrame, video_number: int) -> pd.DataFrame:
    """
    Gets a DataFrame of coordinates and turns this into features.
    In this case, the standard deviation of movement vertically. Extension to also horizontally can be easily made in case this helps for discovering speed.

    :param video_number: The number assigned to 'video' in coord_df
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
    feature_df['Forward_leaning'] = forward_leaning(coord_df)

    return feature_df


def forward_leaning(coord_df: pd.DataFrame) -> List:
    """
    Create forward leaning feature to be used in classification. The forward leaning feature discribes to what extent a person
    leans forward. which could be an indicator of a good runner

    :param coord_df:  A dataframe containing all relevant coördiantes observed in the video.
    :return return_list: returns a list containing containing the absoluut distance that is leaned forward
    """
    fragments = set(coord_df.Fragment)
    return_list = []

    for i in range(len(fragments)):
        fragment_df = coord_df[coord_df['Fragment'] == i + 1]
        shoulder_df = fragment_df[fragment_df['Point'] == 'Right Shoulder']
        hip_df = fragment_df[fragment_df['Point'] == 'Right Hip']
        frames = set(fragment_df.Frame)
        temp_sum = 0
        frame_count = 0
        for j in range(len(frames)):
            difference = shoulder_df.iloc[j, 3] - hip_df.iloc[j, 3]
            # couldn't think of a smarter way to not take the nan values into account for the average
            if difference > 1:
                frame_count += 1
                temp_sum += difference
            if difference < -1:
                frame_count += 1
                temp_sum += difference
        return_list.append(abs(temp_sum / frame_count))
    return return_list
