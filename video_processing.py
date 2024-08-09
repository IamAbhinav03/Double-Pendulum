import cv2
import threading
import queue
import imutils
import numpy as np
from collections import deque
from tools.fps_counter import FPS
import time
import copy
from typing import Tuple, List, Optional

class Coordinate:
    """
    Represents a coordinate with x, y, and time information.
    """
    def __init__(self, x: float, y: float, time: float):
        self.x = x
        self.y = y
        self.time = time

    def __str__(self):
        return f"({self.x}, {self.y}), {self.time}"
    


class VideoCaptureThread(threading.Thread):
    """
    A thread that captures frames from a video source and puts them in a queue.
    If t is not None, stops capturing after 't' seconds.
    """
    def __init__(self, source: str, frame_queue: queue.Queue, t: float | None):
        super().__init__(name="VideoCaptureThread")
        self.source = source
        self.frame_queue = frame_queue
        self.cap = cv2.VideoCapture(source)
        self.stopped = False
        self._num_frames: int = 0
        self.t = t  # Duration to capture
        self.start_time = None

    def run(self):
        """
        Continuously reads frames from the video source and puts them in the frame queue.
        Stops after 't' seconds.
        """
        print(f"{threading.current_thread().name} started.")
        self.start_time = time.time()

        while not self.stopped:

            if self.t is not None:
                current_time = time.time()
                elapsed_time = current_time - self.start_time

                if elapsed_time > self.t:
                    print(f"{threading.current_thread().name}: Reached capture time limit of {self.t} seconds.")
                    self.stop()
                    break

            ret, frame = self.cap.read()
            if not ret:
                print(f"{threading.current_thread().name}: No more frames or error.")
                self.stop()
                break

            self.frame_queue.put(frame)
            self._num_frames += 1

        print(f"{threading.current_thread().name} stopped.")

    def stop(self):
        """
        Stops the video capture thread.
        """
        self.stopped = True
        print(f"CaptureThread processed {self._num_frames} frames.")
        if self.cap.isOpened():
            self.cap.release()

class FrameProcessorThread(threading.Thread):
    """
    A thread that processes frames from the frame queue and puts the processed frames in another queue.
    """
    def __init__(
        self,
        frame_queue: queue.Queue,
        processed_frame_queue: queue.Queue,
        blue_lower: Tuple[int, int, int],
        blue_upper: Tuple[int, int, int]
    ):
        super().__init__(name="FrameProcessorThread")
        self.frame_queue = frame_queue
        self.processed_frame_queue = processed_frame_queue
        self.stopped = False
        self.blue_lower = blue_lower
        self.blue_upper = blue_upper
        self._frame: Optional[np.ndarray] = None
        self._origin: Optional[Tuple[int, int]] = None
        self._coordinate_array: List[Coordinate] = []
        self._display_frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._contours: List[np.ndarray] = []
        self._pts: deque = deque(maxlen=256)
        self.num_processed_frames: int = 0

    def run(self):
        """
        Continuously processes frames from the frame queue and puts the processed frames in the processed_frame_queue.
        """
        print(f"{threading.current_thread().name} started.")
        while not self.stopped:
            if not self.frame_queue.empty():
                # print(f"{threading.current_thread().name} getting frame to frame_queue")
                self._frame = self.frame_queue.get()
                self._process()
                # print(f"{threading.current_thread().name} putting frame to processed_queue")
                self.processed_frame_queue.put(self._display_frame)
            # else:
                # print(f"frame_queue empty")
        print(f"{threading.current_thread().name} stopped.")

    def _preprocess_frame(self):
        """
        Preprocesses the frame, including resizing, blurring, and converting to HSV color space.
        """
        if self._frame is not None:
            self._frame = imutils.resize(self._frame, width=600)
            self._frame = self._frame[:, 200:600]
            self._display_frame = copy.deepcopy(self._frame)
            self._frame = cv2.GaussianBlur(self._frame, (17, 17), 0)
            self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)

            # Calculate the center of the frame (origin)
            frame_height, frame_width = self._frame.shape[:2]
            self._origin = (frame_width // 2, frame_height // 2)

            # Draw marker for origin
            cv2.drawMarker(self._display_frame, self._origin, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    def _create_mask(self):
        """
        Creates a mask for the blue color range.
        """
        if self._frame is not None:
            self._mask = cv2.inRange(self._frame, self.blue_lower, self.blue_upper)
            self._mask = cv2.erode(self._mask, None, iterations=2)
            self._mask = cv2.dilate(self._mask, None, iterations=2)

    def _find_contours(self):
        """
        Finds the contours in the mask.
        """
        if self._mask is not None:
            contours = cv2.findContours(self._mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._contours = imutils.grab_contours(contours)

    def _draw_tracking_circle(self):
        """
        Draws a circle around the largest blue object and calculates its coordinates relative to the frame origin.
        """
        if len(self._contours) > 0:
            largest_contour = max(self._contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            moments = cv2.moments(largest_contour)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

            if radius > 1:
                if self._frame is not None:
                    cv2.circle(self._display_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(self._display_frame, center, 5, (0, 0, 255), -1)
                    # Calculate coordinates relative to origin
                    x = center[0] - self._origin[0]
                    y = self._origin[1] - center[1]  # Invert Y-axis
                    self._coordinate_array.append(Coordinate(x=x, y=y, time=time.time()))

            self._pts.appendleft(center)

    def _draw_path(self):
        """
        Draws the path of the tracked object.
        """
        for i in range(1, len(self._pts)):
            if self._pts[i - 1] is None or self._pts[i] is None:
                continue
            thickness = int(np.sqrt(len(self._pts) / float(i + 1)) * 2.5)
            if self._frame is not None:
                cv2.line(self._display_frame, self._pts[i - 1], self._pts[i], (0, 0, 255), thickness)

    def _process(self):
        """
        Processes the frame, including preprocessing, masking, contour detection, and drawing the tracking circle and path.
        """
        self.num_processed_frames += 1
        self._preprocess_frame()
        self._create_mask()
        self._find_contours()
        self._draw_tracking_circle()
        self._draw_path()

    def stop(self):
        """
        Stops the frame processor thread.
        """
        self.stopped = True
        print(f"FrameProcessorThread processed {self.num_processed_frames} frames.")

    def get_coordinates(self):
        return self._coordinate_array

def main():
    """
    Main function that sets up the video capture and frame processing threads and runs the main loop.
    """
    source = "sample.mp4"
    # source = 0
    blue_lower = (102, 41, 2)
    blue_upper = (179, 255, 255)
    frame_queue = queue.Queue(maxsize=10)
    processed_frame_queue = queue.Queue(maxsize=10)

    video_capture_thread = VideoCaptureThread(source, frame_queue, t=None)
    frame_processor_thread = FrameProcessorThread(frame_queue, processed_frame_queue, blue_lower, blue_upper)

    fps = FPS()
    fps.start()

    video_capture_thread.start()
    frame_processor_thread.start()

    try:
        while True:
            if not processed_frame_queue.empty():
                processed_frame = processed_frame_queue.get()
                cv2.imshow("Processed Frame", processed_frame)
                fps.update()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('stop')
                    fps.stop()
                    # video_capture_thread.stop()
                    # frame_processor_thread.stop()
                    break
            else:
                # print(f"processed_queue empty")
                if video_capture_thread.stopped and frame_processor_thread.frame_queue.empty() and processed_frame_queue.empty():
                    print("No more frames to process. Exiting.")
                    frame_processor_thread.stop()
                    break
                else:
                    time.sleep(0.01)  # Reduce CPU usage while waiting for frames

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping video_capture_thread")
        video_capture_thread.stop()
        print("Stopping frame_processor_thread")
        frame_processor_thread.stop()
        fps.stop()
        video_capture_thread.join()
        frame_processor_thread.join()
        cv2.destroyAllWindows()
        print(f"Final FPS: {fps.fps()}")
        print(f"Total frames processed: {fps._num_frames}")
        # coordinates = 
        print(f"Coordinates\n {len(frame_processor_thread.get_coordinates())}")