import os

from models.config import Config


def get_list_videos(vid_dir):
    """
    Get all video file names

    :param vid_dir: Directory with all videos

    :return: List containing all video file names
    """
    f = []
    for (dir_path, dir_names, file_names) in os.walk(vid_dir):
        f.extend(file_names)
    return f


def run_openpose(config: Config):
    """
    Run openpose for all videos in the video directory from the config.json file

    :param config: Instantiation of the Config class
    """
    wd = os.getcwd()
    os.chdir(config.openpose)
    for video in get_list_videos(config.video_location):
        os.system(
            r'bin\OpenPoseDemo.exe --video "{0}\{1}" --write_json "{2}\{1}"'.format(config.video_location, video,
                                                                                    config.openpose_output))
    os.chdir(wd)

if __name__ == '__main__':
    # set the local files where the video's and openpose output are stored
    run_openpose(Config.get_config())
