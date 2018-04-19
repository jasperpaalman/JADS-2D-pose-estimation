import os

from models.config import Config


def get_list_videos(vid_dir):
    f = []
    for (dir_path, dir_names, file_names) in os.walk(vid_dir):
        f.extend(file_names)
    return f


def run_openpose(video_root, openpose_output_root, openpose_root):
    os.chdir(openpose_root)
    for video in get_list_videos(video_root):
        os.system(
            r'bin\OpenPoseDemo.exe --video "{0}\{1}" --write_json "{2}\{1}"'.format(video_root, video,
                                                                                    openpose_output_root))


if __name__ == '__main__':
    # set the local files where the video's and openpose output are stored

    config = Config.get_config()
    run_openpose(config.video_location, config.openpose_output, config.openpose)
