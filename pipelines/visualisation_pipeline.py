import warnings
import os
import numpy as np
from itertools import zip_longest
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from itertools import chain
from models.config import Config
from models.video import Video
from models.preprocessor import Preprocessor

warnings.filterwarnings('ignore')

class Visualisation:
    """
    Visualisation class to be able to use the matplotlib.animation.FuncAnimation with global variables and keep it
    reasonably structured.
    """
    def __init__(self):
        self.connections = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
            (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
            (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
        ]

        # Preset figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 10))

        # Instantiate points, lines and annotation for the plot
        self.points, = self.ax.plot([], [], 'o')
        self.lines = [self.ax.plot([], [])[0] for i in range(len(self.connections))]
        self.annotation = self.ax.annotate('', xy=(0.02, 0.95), xycoords='axes fraction',
                                 bbox=dict(facecolor='red', alpha=0.5), fontsize=12)

    def process_data(self, clip_name) -> Preprocessor:
        """
        Process data and return preprocessor instance
        :param clip_name: Clip name with at the end .mp4 which will be fetched from the video_data folder
        :return preprocessor: Instance of the Preprocessor class from which attributes can be directly retrieved
        """

        config: Config = Config.get_config()

        folder_name = config.video_data
        video_data_file = ''.join(clip_name.split('.')[:-1]) + '.json'
        video = Video.from_json(os.path.join(folder_name, video_data_file))

        # Convert to usable data type period_running_person division, alle fragment soorten
        preprocessor = Preprocessor(video)

        return preprocessor

    def get_plottables(self, period_person_division, running_person_identifiers, running_fragments, turning_fragments):
        """
        Function to construct all plottable files. In principle to be used for visualisation.

        :param period_person_division: Dictionary within dictionary of the coordinates for each person for each frame
        :param running_person_identifiers: Set of integers, indicating the running people
        :param running_fragments: List of tuples with a start and end frame for each running fragment
        :param turning_fragments: List of tuples with a start and end frame for each turning fragment
        :return:
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

    def func_init(self):
        """
        Initialize function which will be called to create the base frame upon which the animation takes place.
        This is used for blitting to create smoother animations
        :return: Tuple with all plottable objects
        """
        self.points.set_data([], [])
        for line in self.lines:
            line.set_data([],[])
        self.annotation.set_text('')

        return tuple(self.lines) + (self.points, self.annotation)

    def set_axis_limits(self, plottables, image_h, image_w, zoom, pad):
        if zoom:
            y_coords = np.array([coords[~(coords == 0).any(axis=1)][:, 1]
                                 for period_dictionary in plottables.values() for coords in period_dictionary.values()])

            y_coords = np.array(list(chain.from_iterable(y_coords))) + image_h

            cy = np.mean(y_coords)  # y center
            stdy = np.std(y_coords)  # y standard deviation

            self.ydiff = stdy * pad * 2  # total range of y

            self.ax.set_ylim(cy - stdy * pad, cy + stdy * pad)  # set y-limits by padding around the average center of y

            # self.ax.set_xticks([])
            # self.ax.set_yticks([])
        else:
            self.ax.set_ylim([0, image_h])
            self.ax.set_xlim([0, image_w])

    def plot_person(self, frame, plottables, image_h, image_w, zoom=True, pad=3):
        """
        Function that is used by matplotlib.animation.FuncAnimation to iteratively plot given a frame
        :param frame: Frame to plot
        :param frame_to_index: Dictionary with frame as key and index through enumeration as value
        :param plottables: Dictionary within dictionary of the coordinates for a person for each frame
        :param image_h: Video height
        :param image_w: Video width
        :param zoom: Boolean indicating whether or not the animation is zoomed in
        :param pad: Float/integer indicating what the padded region around the animated person should be
        :return: Tuple with all plottable objects
        """

        for person in plottables[frame].keys():
            plot_coords = plottables[frame][person]
            plot_coords[:, 1] = plot_coords[:, 1] + image_h

            coord_dict = {key: value for key, value in dict(enumerate(plot_coords[:, :2])).items() if
                          not (value == 0).any()}

            present_keypoints = set(coord_dict.keys())

            present_connections = [connection for connection in self.connections if
                                   len(present_keypoints & set(connection)) == 2]

            plot_lines = [np.transpose([coord_dict[a], coord_dict[b]]) for a, b in present_connections]

            for coords, line in zip_longest(plot_lines, self.lines):
                if isinstance(coords, np.ndarray):
                    line.set_data(coords[0],coords[1])
                else:
                    line.set_data([],[])

            plot_coords = plot_coords[~(plot_coords == 0).any(axis=1)]

            self.points.set_data(plot_coords[:, 0], plot_coords[:, 1])

            self.annotation.set_text('Frame: {}'.format(frame))

            self.ax.set_xlabel('X coordinate')
            self.ax.set_ylabel('Y coordinate')

            if zoom:
                aspect = image_w / image_h
                xlow, xhigh = plot_coords[:, 0].min(), plot_coords[:, 0].max()  # get x higher and lower limit
                xdiff = xhigh - xlow  # calculate the total range of x
                xpad = ((self.ydiff * aspect) - xdiff) / 2  # calculate how much the xlimits should be padded on either side to set aspect ratio correctly
                self.ax.set_xlim(xlow - xpad, xhigh + xpad)  # set new limits

            break

        return tuple(self.lines) + (self.points, self.annotation)

    def run_animation(self, clip_name, fragment, zoom=True, pad=3, interval=100):
        """
        Complete pipeline to process and plot data.

        :param clip_name: Clip name with at the end .mp4 which will be fetched from the video_data folder
        :param fragment: String indicating what part should be visualised (run, turn or all)
        :return: None
        """

        preprocessor = self.process_data(clip_name)

        period_person_division = preprocessor.period_person_division
        running_person_identifiers = preprocessor.get_running_person_identifiers()
        running_fragments = preprocessor.get_running_fragments()
        turning_fragments = preprocessor.get_turning_fragments()

        period_running_person_division, running_plottables, turning_plottables = self.get_plottables(period_person_division, running_person_identifiers, running_fragments, turning_fragments)

        if fragment == 'run':
            plottables = running_plottables
        elif fragment == 'turn':
            plottables = turning_plottables
        else:
            plottables = period_running_person_division

        self.set_axis_limits(plottables, preprocessor.height, preprocessor.width, zoom=zoom, pad=pad)

        animate = animation.FuncAnimation(fig=self.fig, func=self.plot_person, frames=plottables.keys(), fargs=(plottables,
                        preprocessor.height, preprocessor.width, zoom, pad), interval=interval, init_func=self.func_init, blit=False, repeat=False)
        plt.show()

if __name__ == '__main__':
    visualisation = Visualisation()
    visualisation.run_animation(clip_name='jeroenkrol_28121995_184_80_16500.mp4', fragment='run', zoom=True, pad=3, interval=100)

