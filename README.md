# Double-Pendulum
Code to track the movements of a double pendulum

# How to use
* Clone the repo
* Create a python virtual environment
    ```
    python3 -m venv venv
    ```
* Install the dependecies from the requirements.txt file
    ```
    pip install -r requirements.txt
    ```
* To run the tracking algorithm in **real time**, using webcam run
    ```
    python3 tracking.py
    ```

    * To run the tracking algorithm in an video file, run the previous command with the -v or --video flag.

    ```
    python3 tracking.py -v "path/to/the/video/file"
    ```

    * Add the -b or --buffer flag to the previous command to control the number of tracking values. Default buffer value is 64, eg: 

    ```
    python3 tracking.py -b 128
    ```

    * Use p to pause and play and q to quit.



