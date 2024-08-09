# Double-Pendulum

Code to track the movements of a double pendulum

# How to use

- Clone the repo
- Create a python virtual environment
  ```
  python3 -m venv venv
  ```
- Install the dependecies from the requirements.txt file
  ```
  pip install -r requirements.txt
  ```
- To run the tracking algorithm in **real time**, using webcam run

  ```
  python3 tracking.py
  ```

  - To run the tracking algorithm in an video file, run the previous command with the -v or --video flag.

  ```
  python3 tracking.py -v "path/to/the/video/file"
  ```

  - Add the -b or --buffer flag to the previous command to control the number of tracking values. Default buffer value is 64, eg:

  ```
  python3 tracking.py -b 128
  ```

  - Use p to pause and play and q to quit.

  # Instructions for dealing with files in test branch

  - This branch is meant to test out different ideas to make the tracking better.

  - Right now there is a much better tracking code in test.py, but it fails in live stream perfomance only giving around 20fps meanwhile it give around 80fps on video files.

  # TODOS

  - Improve test.py to work better on video file and live stream

  - Add a feature which will enable users to clip a live stream and get cordinates from any n frames of the video

  - Refactor the code to make it modular
