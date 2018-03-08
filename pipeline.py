from itertools import chain
import matplotlib.pyplot as plt
import cv2
import methods
import pickle
import numpy as np
import plotly.plotly as py

cam = cv2.VideoCapture(r'C:\Users\jaspe\tf-openpose\clips\20180205_182104.mp4')
ret_val, image = cam.read()
image_h, image_w = image.shape[:2]  # getting clip resolution using opencv
fps = cam.get(cv2.CAP_PROP_FPS)  # getting clip frames per second using opencv

with open('people_per_file_clip_20180205_185116.pickle', 'rb') as file:
    people_per_file = pickle.load(file)

people_per_file = people_per_file | methods.get_openpose_output(
    r'C:\Users\jaspe\tf-openpose\demo\openpose-1.2.1-win64-binaries\coordinates\20180205_182104')

print(people_per_file[0])
# Manually set resolution if its not there
image_h = image_h | 1080
image_w = image_w | 1920
fps = fps | 30

connections = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17), (2, 16), (5, 17)
]

# Todo: replace none with sensible stuff
normalized_moved_distance_per_person = methods.normalize_moved_distance_per_person(None)

# Only include identified people that move more than a set movement threshold
maximum_normalized_distance = max(normalized_moved_distance_per_person.values())
movement_threshold = maximum_normalized_distance / 4
moving_people = [key for key, value in normalized_moved_distance_per_person.items() if value > movement_threshold]

plottables_per_file, period_person_division = methods.get_plottables_per_file_and_period_person_division(
    people_per_file, fps, connections)

mean_x_per_person = methods.get_mean_x_per_person(period_person_division)

mean_x_per_moving_person = {key: np.array([[period, x] for period, x in value.items()])
                            for key, value in mean_x_per_person.items() if key in moving_people}

person_plottables_df = methods.get_person_plottables_df(mean_x_per_person, moving_people)

dbscan_subsets = methods.get_dbscan_subsets(maximum_normalized_distance, person_plottables_df)

linked_people = methods.get_linked_people(maximum_normalized_distance,
                                          methods.get_links(moving_people, mean_x_per_moving_person))

plottable_subsets = dbscan_subsets + linked_people

person_period_division = methods.get_person_period_division(period_person_division)
plottable_people = plottable_subsets[
    np.argmax([sum([len(person_period_division[person]) for person in subset]) for subset in plottable_subsets])]

turning_point_index = person_plottables_df[person_plottables_df['Person'].isin(plottable_people)]['X mean'].argmin()

turning_point = person_plottables_df.loc[turning_point_index, 'Period']

person_plottables = [{person: coords for person, coords in period_dictionary.items() if person in plottable_people}
                     for period, period_dictionary in period_person_division.items()]

person_plottables = list(filter(lambda x: x != {}, person_plottables))

f, ax = plt.subplots(figsize=(14, 10))
xspeed = 4

for t in range(len(person_plottables)):
    methods.plot_person(person_plottables[t], f, ax, connections)

coord_list = methods.prepare_data_for_plotting(period_person_division, plottable_people, turning_point)

coord_df = methods.get_dataframe_from_coords(coord_list)

coord_df.to_csv('coordinate_df')

py.tools.set_credentials_file(username='colinvl', api_key='1OPZLs5vGngi8R4dDulM')

pointlist = coord_df.Point.value_counts().index.tolist()
