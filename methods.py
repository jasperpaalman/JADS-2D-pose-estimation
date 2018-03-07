import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import plotly as py
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

# pip install .whl file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# pip install numpy --upgrade if numpy.multiarray error

from itertools import chain
from itertools import groupby

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt


def get_openpose_output(location):
	people_per_file = []
	# coordinate files are ordered, so we can iterate through the folder in which the coordinates are stores for a clip
	# each file corresponds to a frame
	for path, subdirs, files in os.walk(location):
		for filename in files:
			coord_path = os.path.join(path, filename)
			with open(coord_path) as f:
				people_per_file.append(json.load(f)['people'])

	return people_per_file


def rmse(a, b):
	return sqrt(mean_squared_error(a, b))


# Getting plottable information per file
def get_plottables_per_file(people_per_file, period, file, fps, connections):
	plottables_per_file = []  # used for plotting all the coordinates and connected body part lines

	# for each period all 'people' are stored. For a certain period this will allow us to look back in time at the previous x frames
	# in order to be able to group people in disjoint frames together

	period_person_division = {}
	next_person = 0  # used to create a new person when the algorithm can't find a good person fit based on previous x frames

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

			### Identifying people over disjoint frames ###

			best_person_fit = None  # Initially no best fit person in previous x frames is found
			if period != 0:  # period == 0 means no identified people exist, so we need to create them ourselves
				min_rmse = 1000  # set sufficiently high rmse so it will be overwritten easily
				used_joints = list(set(range(18)) - empty_joints)  # only select used joints
				cx = np.mean(person_coords[used_joints, 0])  # center x-coordinate
				cy = np.mean(person_coords[used_joints, 1])  # center y-coordinate
				# set rmse_threshold equal to the mean distance of each used joint to the center
				rmse_threshold = np.mean(pairwise_distances(np.array((cx, cy)).reshape(1, 2),
				                                            person_coords[used_joints, :2]))

				max_frame_diff = int(
					fps // 4)  # number of frames to look back, set to 0.25 sec rather than number of frames
				if period < max_frame_diff:
					j = period
				else:
					j = max_frame_diff

				for i in range(1, j + 1):  # for all possible previous periods within max_frame_diff
					for earlier_person in period_person_division[period - i].keys():  # compare with all people
						if earlier_person not in period_person_division[
							period].keys():  # if not already contained in current period
							earlier_person_coords = period_person_division[period - i][earlier_person]
							empty_joints_copy = empty_joints.copy()
							empty_joints_copy = empty_joints_copy | set(
								np.where((earlier_person_coords == 0).all(axis=1))[0])
							used_joints = list(set(range(18)) - empty_joints_copy)
							if len(used_joints) == 0:
								continue
							# compute root mean squared error based only on mutual used joints
							person_distance = rmse(earlier_person_coords[used_joints, :], person_coords[used_joints, :])
							if person_distance < rmse_threshold:  # account for rmse threshold (only coordinates very close)
								if person_distance < min_rmse:  # if best fit, when compared to previous instances
									min_rmse = person_distance  # overwrite
									best_person_fit = earlier_person  # overwrite
				if best_person_fit != None:  # if a best person fit is found
					period_person_division[period][best_person_fit] = person_coords
				else:  # else create new next person
					period_person_division[period][next_person] = person_coords
					next_person += 1
			else:  # create new next people since it is the first period
				period_person_division[period][next_person] = person_coords
				next_person += 1

		### For plotting the entire video ###

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
	return plottables_per_file


def plot_fit(plottables_per_file, period, f, ax):
	plot_coords = plottables_per_file[period]['plot_coords']
	plot_lines = plottables_per_file[period]['plot_lines']

	plt.scatter(x=plot_coords[:, 0], y=plot_coords[:, 1])

	for x, y in plot_lines:
		plt.plot(x, y)

	ax.set_xlim([0, image_w])
	ax.set_ylim([-image_h, 0])

	f.canvas.draw()
	ax.clear()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# In[ ]:


# f, ax = plt.subplots(figsize=(14,10))
# xspeed = 4

# for t in range(0, len(plottables_per_file)):
#     plot_fit(plottables_per_file, period=t, f=f, ax=ax)
# #     time.sleep(1/fps/xspeed)


# ## Extracting person under observation

# *Identifying moving people*

# In[ ]:


# Basically change the layout of the dictionary
# Now you first index based on the person and then you index based on the period

person_period_division = {}
for person in set(chain.from_iterable(period_person_division.values())):
	person_period_division[person] = {}
	for period in period_person_division.keys():
		period_dictionary = period_person_division[period]
		if person in period_dictionary:
			person_period_division[person][period] = period_dictionary[person]

# In[ ]:


# Calculate the mean x-position of a person in a certain period

mean_x_per_person = {person: {period: np.mean(coords[~(coords == 0).any(axis=1), 0])
                              for period, coords in time_coord_dict.items()}
                     for person, time_coord_dict in person_period_division.items()}

# In[ ]:


# Calculate moved distance by summing the absolute difference over periods
# Normalize moved distance per identified person over frames by including the average frame difference and the length
# of the number of frames included

normalized_moved_distance_per_person = {
	person: pd.Series(mean_x_dict).diff().abs().sum() / (
			np.diff(pd.Series(mean_x_dict).index).mean() * len(mean_x_dict))
	for person, mean_x_dict in mean_x_per_person.items()}

# In[ ]:


# Only include identified people that move more than a set movement threshold

maximum_normalized_distance = max(normalized_moved_distance_per_person.values())
movement_threshold = maximum_normalized_distance / 4
moving_people = [key for key, value in normalized_moved_distance_per_person.items() if value > movement_threshold]

# *Finding person under observation based on clustering with DBSCAN*

# In[ ]:


person_plottables_df = pd.DataFrame(
	[(period, person, x) for person, period_dict in mean_x_per_person.items() if person in moving_people
	 for period, x in period_dict.items()], columns=['Period', 'Person', 'X mean'])

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:


pd.DataFrame({key: value for key, value in mean_x_per_person.items() if key in moving_people}).plot()

# In[ ]:


from sklearn.cluster import DBSCAN

db = DBSCAN(eps=maximum_normalized_distance * 2, min_samples=1)

db.fit(person_plottables_df[['Period', 'X mean']])

person_plottables_df['labels'] = db.labels_

maximum_label = person_plottables_df.groupby('labels').apply(len).sort_values(ascending=False).index[0]

# In[ ]:


DBSCAN_subsets = person_plottables_df.groupby('labels')['Person'].unique().tolist()

DBSCAN_subsets = [list(i) for i in DBSCAN_subsets]

# *Supplementing DBSCAN result with person-specific extrapolation and matching based on RMSE*

# In[ ]:


mean_x_per_moving_person = {key: np.array([[period, x] for period, x in value.items()])
                            for key, value in mean_x_per_person.items() if key in moving_people}

# In[ ]:


links = []
for n, person in enumerate(moving_people):
	x = mean_x_per_moving_person[person][:, 0]
	y = mean_x_per_moving_person[person][:, 1]

	# calculate polynomial
	z = np.polyfit(x, y, 1)
	f = np.poly1d(z)

	if n > 0:
		previous_person = moving_people[n - 1]

		previous_x = mean_x_per_moving_person[previous_person][:, 0]
		previous_y = mean_x_per_moving_person[previous_person][:, 1]

		previous_rmse = rmse(f(previous_x), previous_y)

		links.append((previous_person, person, previous_rmse))

	if n < len(moving_people) - 1:
		next_person = moving_people[n + 1]

		next_x = mean_x_per_moving_person[next_person][:, 0]
		next_y = mean_x_per_moving_person[next_person][:, 1]

		next_rmse = rmse(f(next_x), next_y)

		links.append((person, next_person, next_rmse))

# In[ ]:


# Averaging RMSE between links
link_rmse = np.array(
	[(key, np.mean(np.array(list(group))[:, 2])) for key, group in groupby(links, lambda x: (x[0], x[1]))])

# Use threshold on RMSE to get linked people
linked_people = link_rmse[link_rmse[:, 1] < maximum_normalized_distance * 2][:, 0]

# Setting in right format
linked_people = [list(i) for i in linked_people]

# In[ ]:


# Merge lists that share common elements

plottable_subsets = DBSCAN_subsets + linked_people

all_moving_people = set(chain.from_iterable(plottable_subsets))

for each in all_moving_people:
	components = [x for x in plottable_subsets if each in x]
	for i in components:
		plottable_subsets.remove(i)
	plottable_subsets += [list(set(chain.from_iterable(components)))]

# In[ ]:


plottable_people = plottable_subsets[
	np.argmax([sum([len(person_period_division[person]) for person in subset]) for subset in plottable_subsets])]

# In[ ]:


turning_point_index = person_plottables_df[person_plottables_df['Person'].isin(plottable_people)]['X mean'].argmin()

# In[ ]:


turning_point = person_plottables_df.loc[turning_point_index, 'Period']

# ## Plot person under observation

# In[ ]:


person_plottables = [{person: coords for person, coords in period_dictionary.items() if person in plottable_people}
                     for period, period_dictionary in period_person_division.items()]

# In[ ]:


person_plottables = list(filter(lambda x: x != {}, person_plottables))


# In[ ]:


def plot_person(plottables, f, ax):
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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# In[ ]:


f, ax = plt.subplots(figsize=(14, 10))
xspeed = 4

for t in range(len(person_plottables)):
	plot_person(person_plottables[t], f=f, ax=ax)


#     time.sleep(1/fps/xspeed)


# ## Plotting coordinates of joints

# In[ ]:


def prepare_data_for_plotting(period_person_division, plottable_people):
	coord_list = []
	for period, period_dictionary in period_person_division.items():
		for person, coords in period_dictionary.items():
			if person in plottable_people and period < turning_point:
				coord_dict = {key: value for key, value in dict(enumerate(coords[:, :2])).items() if 0 not in value}
				coord_list.append(coord_dict)
				break

	return coord_list


# In[ ]:


coord_list = prepare_data_for_plotting(period_person_division, plottable_people)

# *To dataframe*

# In[ ]:


coord_df = pd.DataFrame(coord_list)

coord_df.columns = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Hand',
                    'Left Shoulder', 'Left Elbow', 'Left Hand',
                    'Right Hip', 'Right Knee', 'Right Foot', 'Left Hip', 'Left Knee', 'Left Foot',
                    'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear']

# add the frame number
coord_df['Frame'] = coord_df.index

# melt the dataframe to get the locations in one row
coord_df = pd.melt(coord_df, id_vars='Frame', var_name='Point', value_name='Location')

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

coord_df.to_csv('coordinate_df')

# coord_df = coord_df[coord_df['x'] <= 1780]

coord_df.head()



# #you need to create an account an get an ID in order to be able to run this
py.tools.set_credentials_file(username='colinvl', api_key='1OPZLs5vGngi8R4dDulM')


# In[ ]:


pointlist = coord_df.Point.value_counts().index.tolist()
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
		range=[-600, -300]
	)
	#     ylim = (-600, 300)
)

fig = dict(data=points, layout=layout)
py.iplot(fig, filename="Open pose runtracker")
# py.offline.iplot(fig, filename = "Open pose runtracker")


# In[ ]:


pointlist = coord_df.Point.value_counts().index.tolist()
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
		range=[-600, -300]
	)
	#     ylim = (-600, 300)
)

fig = dict(data=points, layout=layout)
py.iplot(fig, filename="Open pose runtracker boxplot")
# py.offline.iplot(fig, filename = "Open pose runtracker")
