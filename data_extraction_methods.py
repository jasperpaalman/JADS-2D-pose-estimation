import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time
# import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN

# pip install .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# pip install numpy --upgrade if numpy.multiarray error

from itertools import chain
from itertools import groupby

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt

""""This module is meant to extract the data into usable formates to be used in the other modules"""



connections = [
	(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
	(8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
	(0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]
def get_list_videos(vid_dir):
	"""
	Function determines how many video's are in the provided video directory and returns the names of all functions
	:param vid_dir: Directory of all video's
	:return f: list of video file names
	"""
	f = []
	for (dirpath, dirnames, filenames) in os.walk(vid_dir):
		f.extend(filenames)
	return f


def run_openpose(vid_dir,coord_dir, openpose_location):
	"""
	Function that executes the openpose demo for every video in the provided video directory
	:param vid_dir: Directory of all video's
	:param coord_dir: Directory where coordinates have to be stored
	:param openpose_location: Directory of openpose files
	"""
	os.chdir(openpose_location)
	for video in get_list_videos(vid_dir):
		os.system(r'bin\OpenPoseDemo.exe --video "{0}\{1}" --write_json "{2}\{1}"'.format(vid_dir, video, coord_dir))


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
	Using the coordinates all coordinates of a person identified in a specific frame the 'centre' of this person is calculated

	:param used_joints: List of all joints that are identified by OpenPose for this person
	:param person_coords: Coordinates belonging to person

	:returns cx, xy: centre x and centre y coordinates.
	"""
	cx = np.mean(person_coords[used_joints, 0])  # center x-coordinate
	cy = np.mean(person_coords[used_joints, 1])  # center y-coordinate
	return cx, cy


def determine_rmse_threshold(person_coords, used_joints):
	"""
	Using the centre coordinates of a identified person and the known used joints a rsme treshold is calculated

	:param used_joints: List of all joints that are identified by OpenPose for this person
	:param person_coords: Coordinates belonging to person
	:return rmse_threshold: a float value containing the rsme threshold
	"""
	rmse_threshold = np.mean(pairwise_distances(np.array(calc_center_coords_of_person(person_coords, used_joints)).reshape(1, 2),
	                                            person_coords[used_joints, :2]))
	return rmse_threshold


def amount_of_frames_to_look_back(fps, frame):
	"""
	Function that returns the amoount of frames that need be examined.

	:param fps: number of frames per second in current video
	:param frame: current frame in loop
	:return J: Number of frames to be examined
	"""
	# number of frames to look back, set to 0.25 sec rather than number of frames
	max_frame_diff = int(fps // 4)
	if frame < max_frame_diff:
		j = frame
	else:
		j = max_frame_diff
	return j


def get_period_person_division(people_per_file, fps):
	""""


	:param fps: number of frames per second in current video
	:param people_per_file: List of List of dictionaries. So a list of frames, each frame consists of a list of dictionaries in which
	all identified people in the video are described using the coordinates of the observed joints.

	:return period_person_division: data strucure containing per frame all persons and their
								   corresponding coordinates

	"""

	frame_person_division = {}  # Dict of dicts

	# used to create a new person when the algorithm can't find a good person fit based on previous x frames
	next_person = 0

	for frame, file in enumerate(people_per_file):
		frame_person_division[frame] = {}  # for a frame (period) make a new dictionary in which to store the identified people

		for person in file:
			# information for identifying people over disjoint frames
			person_coords = np.array([[x, -y, z] for x, y, z in np.reshape(person['pose_keypoints'], (18, 3))])

			best_person_fit = None  # Initially no best fit person in previous x frames is found
			if frame == 0:  # frame == 0 means no identified people exist (because first frame), so we need to create them ourselves
				frame_person_division[frame][next_person] = person_coords  # create new next people since it is the first frame
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
				for i in range(1, amount_of_frames_to_look_back(fps, frame) + 1):
					for earlier_person in frame_person_division[frame - i].keys():  # compare with all people
						if earlier_person not in frame_person_division[frame].keys():
							# if not already contained in current period
							earlier_person_coords = frame_person_division[frame - i][earlier_person]
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
					frame_person_division[frame][best_person_fit] = person_coords
				else:  # else create new next person
					frame_person_division[frame][next_person] = person_coords
					next_person += 1
	return frame_person_division


def get_person_period_division(period_person_division):
    """
    Function that reverses the indexing in the dictionary
    :param period_person_division: data strucure containing per frame all persons and their
                                   corresponding coordinates
    :return person_period_division: data structure containing per person all frames and the coordinates
                                    of that person in that frame.
    """
    person_period_division = {}
    for person in set(chain.from_iterable(period_person_division.values())):
        person_period_division[person] = {}
        for period in period_person_division.keys():
            period_dictionary = period_person_division[period]
            if person in period_dictionary:
                person_period_division[person][period] = period_dictionary[person]
    return person_period_division