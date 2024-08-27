# Double-Pendulum

Code to track the movements of a double pendulum


This branch is meant to be a scratch space to test out the server code. **You have to install ffmpeg to use the code with a stream**

**Stuff to do before running the code**

- Clone the repo
- Create a python virtual environment
  ```
  python3 -m venv venv
  ```
- Install the dependecies from the requirements.txt file
  ```
  pip install -r requirements.txt
  ```


**Setting up the ffmpeg sever**
```
ffmpeg -stream_loop -1 -re -i <path_to_your_video_file/device> -preset ultrafast -vcodec libx264 -tune zerolatency -b:v 900k -f h264 udp://127.0.0.1:5000\?overrun_nonfatal=1
```

**The server code is the folder random_number_generator**

**How to run the server**

*Change the source in main.py line 36 to use other sources ```generator.start_processing('udp://127.0.0.1:5000/')```*

- Navigate to the /random_number_generator/backend/ on a new termainal
- Run for development
  ```
  fastapi dev
  ```
- Run for production
  ```
  fastapi run main
  ```


# Known Bugs

**The code is crashing due to a memeory allocation error, especially when large n digit number is being generated**



# How to use the tracking code only


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
