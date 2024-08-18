import threading
import queue
import cv2
import time 
import imutils
import numpy as np
from typing import Optional
from collections import deque

class Coordinate:
    """
    Represents a coordinate with x, y and time information.
    """

    def __init__(self, x: float, y: float, time: float):
        self.x = x
        self.y = y
        self.time = time

    def __str__(self):
        return f"({self.x}, {self.y}), {self.time}"
    

class VideoCaptureThread(threading.Thread):
    def __init__(self, source):
        super().__init__(name="VideoCaptureThread")
        self._source = source
        self._frames_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self._cap = cv2.VideoCapture(self._source)
        self.exception = None
        time.sleep(2.0)
        print("Initialized Capture object")

    def run(self):
        print("Running capture thread")
        try:
            if not self._cap.isOpened():
                raise RuntimeError("Error starting capture device")

            while not self.stop_event.is_set():
                ret, frame = self._cap.read()
                if ret:
                    self._frames_queue.put(frame)
                else:
                    print("Issue grabbing frame, restarting capture device")
                    self._cap = cv2.VideoCapture(self._source)
                    # self.stop()
                    # break

        except Exception as e:
            self.exception = e
            self.stop()

    def stop(self):
        self.stop_event.set()
        if self._cap.isOpened():
            self._cap.release()

    def get_exception(self):
        return self.exception


class FrameProcessorThread(threading.Thread):
    def __init__(self, frames_queue):
        super().__init__(name="FrameProcessorThread")
        self.frames_queue = frames_queue
        self.bits_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.exception = None

    def run(self):
        print("Running frame processor thread")
        try:
            while not self.stop_event.is_set():
                if not self.frames_queue.empty():
                    print("Fetching frames from frame queue")
                    frame = self.frames_queue.get()
                    bits = self._process(frame)
                    print(f"bits: {bits}")
                    if len(bits) > 0:
                        print("Inserting bits to the bits queue")
                        self.bits_queue.put(bits)
                else:
                    time.sleep(0.1)  # Avoid busy waiting

        except Exception as e:
            self.exception = e
            self.stop()

    def _process(self, frame):
        # Image processing steps go here
        # Example process:
        frame = imutils.resize(frame, width=600)
        frame = frame[:, 200:600]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Example bit extraction logic
        mask = cv2.inRange(frame, (102, 41, 2), (179, 255, 255))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(largest_contour)
            return f"{int(x)}{int(y)}"
        return "00"

    def stop(self):
        self.stop_event.set()

    def get_exception(self):
        return self.exception



    
