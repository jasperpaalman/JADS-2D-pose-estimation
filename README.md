# JADS 2D pose estimation
This repository will be used to analyse sprinting and suggest improvements in techniques.

The code for the website is hosted at [Luttik / pro coach for all](https://github.com/Luttik/pro_coach_for_all)

## Setup
The advised way to setup your repository is as follows:

- Clone this repo with `git clone` or your git GUI
- Create a new virtual environment in working directory `virtualenv env` with python version 3.6 (if virualenv is not installed run
`pip install virtualenv` for more information look [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/#lower-level-virtualenv))
- Activate the environment `env\scripts\activate`
- install packages `pip install -r requirements.txt`
- download and setup the open_pose library (version 1.2.1) in the same working directory
    - [Download link](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)
    - Unpack zip in working directory
- Setup config.json using Example_config.json
    - Set "openpose_output" to the directory where you want to store the raw openpose output
    - Set "video_location" to the directory of the video files you want to analyze
    - Set "openpose" to the directory of the openpose binaries on your system
    - Set "video_data" to the directory where you want to store the relevant video data used for analysis
    - Use full paths not abbreviated ones (i.e. C:\WorkingDirectory\OpenPose not .\OpenPose)
  

## Use
If you use a virtual environment (which is highly advised to maintain the correct versions of dependencies) ensure that 
you also run `env\scripts\activate` every time before you start to work (otherwise your dependencies won't load).

This package consists of 2 elements:
- First a pipeline which extracts all data from the video files
- Second a notebook file used to analyze the data

To analyze the video data the multivideo pipeline needs to be executed first this generates all relevant data needed for 
analyzing the data. If the suggested setup method is used this should be fairly easy. This pipeline can be found in the 
pipelines directory. Execute the multivideo pipeline using the python IDE of your choice.

Using the notebook all relevant visualizations can be generated.