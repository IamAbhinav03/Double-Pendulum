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
    Represents a coordinate with x, y, and time information.
    """

    def __init__(self, x: float, y: float, time: float) -> None:
        self.x = x  # The x-coordinate
        self.y = y  # The y-coordinate
        self.time = time  # The time when the coordinate was captured

    def __str__(self) -> str:
        return f"({self.x}, {self.y}), {self.time}"  # String representation of the coordinate


class VideoCaptureThread(threading.Thread):
    """
    A thread that captures video frames from a source and adds them to a queue.
    """

    def __init__(self, source: str) -> None:
        super().__init__(name="VideoCaptureThread")
        self._source: str = source  # The video source (e.g., a stream URL)
        self._frames_queue: queue.Queue = queue.Queue(maxsize=10)  # Queue to store captured frames
        self.stop_event: threading.Event = threading.Event()  # Event to signal when to stop the thread
        self._cap: cv2.VideoCapture = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)  # Video capture object
        self.exception: Optional[Exception] = None  # Stores any exceptions that occur
        time.sleep(5.0)  # Wait for 5 seconds to ensure everything is ready
        print("Initialized Capture object")

    def run(self) -> None:
        """
        Continuously captures video frames and adds them to the frames queue.
        Stops if an error occurs or the stop event is set.
        """
        print("Running capture thread")
        print(f"{__name__}: STOP_EVENT: {self.stop_event.is_set()}")
        if not self.stop_event.is_set():
            try:
                if not self._cap.isOpened():  # Check if the video capture opened successfully
                    raise RuntimeError("Error starting capture device")

                ret, frame = self._cap.read()  # Read a frame from the video source
                if ret:  # If the frame was read successfully
                    self._frames_queue.put(frame)  # Add the frame to the queue
                else:
                    print("Issue grabbing frame, restarting capture device")
                    self._cap.release()  # Release the current video capture
                    self._cap = cv2.VideoCapture(self._source)  # Reopen the video capture

            except Exception as e:
                self.exception = e  # Store any exception that occurs
                self.stop()  # Stop the thread if an error occurs

    def stop(self) -> None:
        """
        Stops the video capture thread and releases the video capture device.
        """
        print("Stopping Video Capture thread")
        self.stop_event.set()  # Signal that we want to stop the thread
        if self._cap.isOpened():  # If the video capture is still open
            print("Releasing capture object")
            try:
                self._cap.release()  # Release the video capture device
                print("Capture device released")
            except Exception as e:
                print("Error releasing capture device")
                raise RuntimeError(f"{e}")
        else:
            print("Capture device is not open")

    def get_exception(self) -> Optional[Exception]:
        """
        Returns any exception that occurred during the thread's execution.
        """
        return self.exception


class FrameProcessorThread(threading.Thread):
    """
    A thread that processes video frames to extract bits and adds them to a queue.
    """

    def __init__(self, frames_queue: queue.Queue) -> None:
        super().__init__(name="FrameProcessorThread")
        self.frames_queue: queue.Queue = frames_queue  # Queue of frames to process
        self.bits_queue: queue.Queue = queue.Queue(maxsize=10)  # Queue to store the extracted bits
        self.stop_event: threading.Event = threading.Event()  # Event to signal when to stop the thread
        self.exception: Optional[Exception] = None  # Stores any exceptions that occur

    def run(self) -> None:
        """
        Continuously processes frames to extract bits and adds them to the bits queue.
        Stops if an error occurs or the stop event is set.
        """
        print("Running frame processor thread")
        try:
            while not self.stop_event.is_set():  # Keep processing frames until we are told to stop
                if not self.frames_queue.empty():  # If there are frames to process
                    print("Fetching frame from frame queue")
                    frame = self.frames_queue.get()  # Get a frame from the queue
                    bits = self._process(frame)  # Process the frame to extract bits
                    if len(bits) > 0:  # If we extracted some bits
                        print("Inserting bits into the bits queue")
                        self.bits_queue.put(bits)  # Add the bits to the bits queue
                else:
                    time.sleep(0.1)  # Avoid busy waiting by sleeping for 100ms

        except Exception as e:
            self.exception = e  # Store any exception that occurs
            self.stop()  # Stop the thread if an error occurs

    def _process(self, frame: np.ndarray) -> str:
        """
        Processes a video frame to extract bits.
        This is a sample implementation that performs some basic image processing
        and returns extracted bits based on pixel locations.
        """
        # Resize the frame to a fixed width
        frame = imutils.resize(frame, width=600)
        # Crop the frame to focus on a region of interest (ROI)
        frame = frame[:, 200:600]
        # Convert the frame from BGR color space to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask to detect a specific color range in the frame
        mask = cv2.inRange(frame, (102, 41, 2), (179, 255, 255))
        mask = cv2.erode(mask, None, iterations=2)  # Erode to remove small noise
        mask = cv2.dilate(mask, None, iterations=2)  # Dilate to restore object size

        # Find contours in the mask
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if contours:  # If we found any contours
            # Find the largest contour and get its center coordinates
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(largest_contour)
            # Return the x and y coordinates as bits (converted to strings)
            return f"{int(x)}{int(y)}"
        return "00"  # Return a default value if no contours were found

    def stop(self) -> None:
        """
        Stops the frame processing thread.
        """
        print("Setting frame processing stop event")
        self.stop_event.set()  # Signal that we want to stop the thread

    def get_exception(self) -> Optional[Exception]:
        """
        Returns any exception that occurred during the thread's execution.
        """
        return self.exception