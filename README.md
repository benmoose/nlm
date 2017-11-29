# Image Processing Coursework

Submission Details

| Student | Email | Student Number |
| ---- | ---- | ---- |
| Ben Hadfield | zchabjh@ucl.ac.uk | 1401907

Implementation Details

| Language | Dependencies |
| ---- | ---- |
| Python | Either [Python 3](https://www.python.org/downloads/), or [Docker](https://www.docker.com/) |

_Note: Docker is the preferred way to run the program, since it doesn't
require you to manually download requirements._

## Non Local Means

### Project Structure

#### Source Code

The source code of my implementation can be found in the `src/` directory.
The algorithm is written in Cython (`.pyx`) which allows performance critical
parts of the model (e.g. nested for loops) to be written in C.

#### Images

The `img/` directory is where the algorithm looks for input images and
writes output images.
Be sure to select an image that is in this directory, otherwise the
program will fail.

The output images are written to `img/out/nlm-<id>.jpg` where `<id>` is
the `id` printed to the console once the program has finished.

The purpose of the `id` is simply to prevent overwriting other images in
the directory.

### Usage

#### Docker
 
 - Unzip the project
 - Navigate to the project root
 - run `docker-compose run py`

#### Python

 - Unzip the project
 - Navigate to the project root
 - Start a virtual environment by running `python3 -m venv venv`
 - Install the project requirements `pip3 install -r requirements.txt`
 - Compile Cython code by running `python setup.py build_ext --inplace`
 - Run the algorithm with `python .`
