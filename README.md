# JADS 2D pose estimation
This repository will be used to analyse sprinting and suggest improvements in techniques.

The code for the website is hosted at [Luttik / pro coach for all](https://github.com/Luttik/pro_coach_for_all)

## Setup
The advised way to setup your repository is as follows:

- Clone this repo with `git clone` or your git GUI
- Create a new virtual environment `virtualenv env` (if virualenv is not installed run `pip install virtualenv` for more information look [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/#lower-level-virtualenv))
- Activate the environment `env\scripts\activate`
- install packages `pip install -r requirements.txt`
- Open the config.json file and change jpjrvp
- download and setup the open_pose library in the same working directory

## Use
If you use a virtual environment (which is highly adviced to maintain the correct versions of dependencies) ensure that you also run `env\scripts\activate` every time before you start to code (otherwise your dependencies won't load)
