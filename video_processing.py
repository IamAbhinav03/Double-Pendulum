import cv2
import imutils
import numpy as np
from threading import Thread
from collections import deque
from typing import Optional, Tuple, Deque


class VideoCaptureThread:
    """
    Class to continuously capture frames from a video source in a separate thread.
    """

    def __init__(self, source: int | str = 0):
        self.stream: cv2.VideoCapture = cv2.VideoCapture(source)
        self.grabbed: bool
        self.frame: Optional[np.ndarray]
        self.grabbed, self.frame = self.stream.read()
        self.stopped: bool = False

    def start(self) -> 'VideoCaptureThread':
        Thread(target=self._capture, name="VideoCaptureThread").start()
        return self

    def _capture(self) -> None:
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def stop(self) -> None:
        self.stopped = True


class VideoProcessor:
    """
    Class to process video frames for detecting and tracking objects.
    """

    def __init__(self, frame: Optional[np.ndarray] = None, buffer_size: int = 64):
        self.frame: Optional[np.ndarray] = frame
        self.buffer_size: int = buffer_size
        self.blue_lower: Tuple[int, int, int] = (102, 41, 2)
        self.blue_upper: Tuple[int, int, int] = (179, 255, 255)
        self.mask: Optional[np.ndarray] = None
        self.contours: list = []
        self.pts: Deque[Optional[Tuple[int, int]]] = deque(maxlen=self.buffer_size)
        self.stopped: bool = False

    def start(self) -> 'VideoProcessor':
        Thread(target=self.process, name="VideoProcessThread").start()
        return self

    def _preprocess_frame(self) -> None:
        """Resize, blur, and convert the frame to HSV color space."""
        if self.frame is not None:
            self.frame = imutils.resize(self.frame, width=600)
            self.frame = cv2.GaussianBlur(self.frame, (17, 17), 0)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

    def _create_mask(self) -> None:
        """Create a mask for the color range and perform morphological operations."""
        if self.frame is not None:
            self.mask = cv2.inRange(self.frame, self.blue_lower, self.blue_upper)
            self.mask = cv2.erode(self.mask, None, iterations=2)
            self.mask = cv2.dilate(self.mask, None, iterations=2)

    def _find_contours(self) -> None:
        """Find contours in the masked frame."""
        if self.mask is not None:
            contours = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours = imutils.grab_contours(contours)

    def _draw_tracking_circle(self) -> None:
        """Draw a circle around the detected object and update tracking points."""
        if len(self.contours) > 0:
            largest_contour = max(self.contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            moments = cv2.moments(largest_contour)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            if radius > 3:
                if self.frame is not None:
                    cv2.circle(self.frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(self.frame, center, 5, (0, 0, 255), -1)

            self.pts.appendleft(center)

    def _draw_path(self) -> None:
        """Draw the path of the detected object."""
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * 2.5)
            if self.frame is not None:
                cv2.line(self.frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

    def process(self) -> None:
        """Process the video frame by applying all steps."""
        self._preprocess_frame()
        self._create_mask()
        self._find_contours()
        self._draw_tracking_circle()
        self._draw_path()

    def stop(self) -> None:
        self.stopped = True



    """
    Class to display video frames in a separate thread.
    """

    def __init__(self, frame: Optional[np.ndarray] = None):
        self.frame: Optional[np.ndarray] = frame
        self.stopped: bool = False

    def start(self) -> 'VideoDisplayThread':
        Thread(target=self.display, name="VideoDisplayThread").start()
        return self

    def display(self) -> None:
        while not self.stopped:
            if self.frame is None or self.frame.size == 0:
                continue
            try:
                cv2.imshow("Video Display", self.frame)
            except cv2.error as e:
                print(f"Error displaying frame: {e}")
            if cv2.waitKey(1) == ord("q"):
                self.stop()

    def stop(self) -> None:
        self.stopped = True