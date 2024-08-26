from collections import deque
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from tools.fps_counter import FPS
from config import PENDULUM_POINTS_BUFFER
from src.seed_generation import generate_seed

class PendulumTracker:
    def __init__(self, args, blue_lower, blue_upper):
        self.args = args
        self.blue_lower = blue_lower
        self.blue_upper = blue_upper
        self.pts = deque(maxlen=args['buffer'])
        self.pendulum_points = deque(maxlen=PENDULUM_POINTS_BUFFER)
        self.unique_points = set()
        self.seed = None
        self.vs = self._init_video_stream()
        self.fps = FPS()
        self.fps.start()

    def _init_video_stream(self):
        if not self.args.get("video", False):
            print("Starting Webcam")
            return VideoStream(src=0).start()
        else:
            print(f"Fetching video from {self.args['video']}")
            return cv2.VideoCapture(self.args["video"])

    def preprocess_frame(self, frame):
        frame = imutils.resize(frame, width=600)
        blurred_frame = cv2.GaussianBlur(frame, (17, 17), 0)
        return cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    def mask_frame(self, hsv_frame):
        mask = cv2.inRange(hsv_frame, self.blue_lower, self.blue_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def find_contours(self, mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)

    def find_min_enclosing_circle(self, cnts, frame):
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 3:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

            self.pts.appendleft(center)
            self.add_unique_point(center)
            return center
        return None

    def add_unique_point(self, point):
        if point not in self.unique_points:
            self.unique_points.add(point)
            self.pendulum_points.append(point)
            if len(self.unique_points) >= 5:
                self.generate_new_seed()

    def generate_new_seed(self):
        unique_points_list = list(self.unique_points)
        self.seed = generate_seed(unique_points_list)
        print(f"New seed generated: {self.seed}")
        self.unique_points.clear()

    def get_current_seed(self):
        return self.seed

    def get_pendulum_points(self):
        return list(self.pendulum_points)

    def draw_locus(self, frame):
        for i in range(1, len(self.pts)):
            if self.pts[i-1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(self.args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i-1], self.pts[i], (0, 0, 255), thickness)

    def process_frame(self):
        frame = self.vs.read()
        frame = frame[1] if self.args.get("video", False) else frame

        if frame is None:
            return None, None

        hsv_frame = self.preprocess_frame(frame)
        mask = self.mask_frame(hsv_frame)
        cnts = self.find_contours(mask)
        center = self.find_min_enclosing_circle(cnts, frame)
        self.draw_locus(frame)

        self.fps.update()
        return frame, center

    def cleanup(self):
        self.fps.stop()
        if not self.args.get("video", False):
            self.vs.stop()
        else:
            self.vs.release()
        print(f"Elapsed time: {self.fps.elapsed():.2f}")
        print(f"Approx. FPS: {self.fps.fps():.2f}")