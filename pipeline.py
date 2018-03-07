import cv2
#import methods
import pickle


# cam = cv2.VideoCapture(r'C:\Users\jaspe\tf-openpose\clips\20180205_182104.mp4')
# ret_val, image = cam.read()
# image_h, image_w = image.shape[:2]  # getting clip resolution using opencv
# fps = cam.get(cv2.CAP_PROP_FPS)  # getting clip frames per second using opencv
#
# people_per_file = \
# 	methods.get_openpose_output(r'C:\Users\jaspe\tf-openpose\demo\openpose-1.2.1-win64-binaries\coordinates\20180205_182104')
people_per_file = None
if people_per_file is None:
	with open('people_per_file_clip_20180205_185116.pickle', 'rb') as file:
		people_per_file = pickle.load(file)
print(people_per_file[0])
# # Manually set resolution if its not there
# image_h = image_h | 1080
# image_w = image_w | 1920
# fps = fps| 30
#
# connections = [
# 	(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
# 	(0, 14), (14, 16),
# 	(0, 15), (15, 17), (2, 16), (5, 17)
# ]


