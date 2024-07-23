from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Initial HSV values for the blue color
blue_lower = (23, 19, 2)
blue_upper = (179, 255, 255)

pts = deque(maxlen=args['buffer'])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    print("Starting Webcam")
    vs = VideoStream(src=0).start()
else:
    print(f"Fetching video from {args['video']}")
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# Initialize variables for playback control
paused = False
frame_counter = 0

# Function to get current trackbar positions
def get_trackbar_values():
    h_lower = cv2.getTrackbarPos('H Lower', 'Control Panel')
    s_lower = cv2.getTrackbarPos('S Lower', 'Control Panel')
    v_lower = cv2.getTrackbarPos('V Lower', 'Control Panel')
    h_upper = cv2.getTrackbarPos('H Upper', 'Control Panel')
    s_upper = cv2.getTrackbarPos('S Upper', 'Control Panel')
    v_upper = cv2.getTrackbarPos('V Upper', 'Control Panel')
    return (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper)

# Create control panel for trackbars
cv2.namedWindow('Control Panel')
cv2.createTrackbar('H Lower', 'Control Panel', blue_lower[0], 179, lambda x: None)
cv2.createTrackbar('S Lower', 'Control Panel', blue_lower[1], 255, lambda x: None)
cv2.createTrackbar('V Lower', 'Control Panel', blue_lower[2], 255, lambda x: None)
cv2.createTrackbar('H Upper', 'Control Panel', blue_upper[0], 179, lambda x: None)
cv2.createTrackbar('S Upper', 'Control Panel', blue_upper[1], 255, lambda x: None)
cv2.createTrackbar('V Upper', 'Control Panel', blue_upper[2], 255, lambda x: None)

while True:
    if not paused or frame_counter == 0:
        # grab the current frame
        og_frame = vs.read()

        # handle the frame from VideoCapture or VideoStream
        og_frame = og_frame[1] if args.get("video", False) else og_frame

        if og_frame is None:
            break

        # resize the frame, blur it, and convert it to the hsv color space
        resized_frame = imutils.resize(og_frame, width=600)
        blurred_frame = cv2.GaussianBlur(resized_frame, (25, 25), 0)
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Get current HSV values from trackbars
        blue_lower, blue_upper = get_trackbar_values()

        # construct a mask for the color "blue", then perform a series of dilations and erosions
        mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 3:
                cv2.circle(resized_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(resized_frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(resized_frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        numpy_horizontal = np.hstack((resized_frame, hsv_frame, mask_frame))
        cv2.imshow("Frames", numpy_horizontal)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("p"):
        cv2.waitKey(-1) 
    elif key == ord("n") and paused:
        frame_counter += 1
        continue

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
