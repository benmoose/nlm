# Image Processing Coursework

###### Submission Details

| Student | Email | Student Number |
| ---- | ---- | ---- |
| Ben Hadfield | zchabjh@ucl.ac.uk | 1401907

###### Implementation Details

| Language | Dependencies | OS
| ---- | ---- | ---- |
| Python 3 | Either [Python 3](https://www.python.org/downloads/) or [Docker](https://www.docker.com/) | Tested on MacOS and Linux |

_Note: Docker is the preferred way to run the program, since it doesn't
require you to manually download requirements._

## Non Local Means

### Project Structure

#### Source Code

The source code of my implementation can be found in the `src/` directory.
The algorithm is written in Cython (`.pyx`) which allows performance critical
parts of the model (e.g. nested for loops) to be written in C.

#### Images

The `img/in/` directory is where the algorithm looks for input images
and writes output images.
Be sure to select an image that is in this directory, otherwise the
program will fail. You can put your own images in here and run the
algorithm on them.

The output images are written to `img/out/<id>.jpg` where `<id>` is the
hexadecimal `id` printed to the console once the program has finished.

The purpose of the `id` is to allow you to quickly identify the
generated image, and to prevent the overwriting of other images in that
directory.

### Usage

#### Docker
 
 - Unzip the project
 - Navigate to the project root
 - run `docker-compose run py`

#### Python

 - Unzip the project
 - Navigate to the project root
 - Create a virtual environment by running `python3 -m venv venv`
 - Activate the virtual environment with `source ./venv/bin/activate`
 - Install the project requirements `pip3 install -r requirements.txt`
 - Compile Cython and run the algorithm with `./run.sh`

### Troubleshooting

_All commands should be executed from the project root._

###### `./run.sh` or `docker-compose run py` gives permission denied

In this case do one of the following, remembering to prepend
`docker-compose run py` to the command if using docker:
 - either run `chmod +x ./run.sh`
 - or run `python3 setup.py build_ext --inplace` followed by `python3 .`
 
###### The algorithm crashes at the end with a `FileNotFoundError`
 
Make sure that there is an `out/` folder in the `img/` directory.
If not then run `mkdir img/out`

###### I'm getting `ModuleNotFoundError: No module named 'src.model'`

Make sure the `src/` dir only contains
 - `__init__.py`
 - `model.pyx`
 
and no other files (e.g. `.c` or `.so` files).
Then try running the algorithm again.
