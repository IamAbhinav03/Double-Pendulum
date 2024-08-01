from collections import deque
import argparse
import time
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import cProfile


def main():
    start_time = time.time()
    global args, blue_lower, blue_upper, pts, frame, hsv_frame, masked_frame  # Declare frame as a global variable

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())

    # HSV mask max and min value
    blue_lower = (102, 41, 2)
    blue_upper = (179, 255, 255)

    # A deque to store the point of length buffer, default 64
    pts = deque(maxlen=args['buffer'])

    # Create control panel for trackbars
    # cv2.namedWindow('Control Panel')
    # cv2.createTrackbar('H Lower', 'Control Panel', blue_lower[0], 179, lambda x: None)
    # cv2.createTrackbar('S Lower', 'Control Panel', blue_lower[1], 255, lambda x: None)
    # cv2.createTrackbar('V Lower', 'Control Panel', blue_lower[2], 255, lambda x: None)
    # cv2.createTrackbar('H Upper', 'Control Panel', blue_upper[0], 179, lambda x: None)
    # cv2.createTrackbar('S Upper', 'Control Panel', blue_upper[1], 255, lambda x: None)
    # cv2.createTrackbar('V Upper', 'Control Panel', blue_upper[2], 255, lambda x: None)

    # if a video path was not supplied, grab the reference to the webcam
    if not args.get("video", False):
        print("Starting Webcam")
        vs = VideoStream(src=0).start()
    else:
        print(f"Fetching video from {args['video']}")
        vs = cv2.VideoCapture(args["video"])

    # allow the camera or video file to warm up
    time.sleep(2.0)
    try:
        print(f"Total no. frames in the video: {vs.get(cv2.CAP_PROP_FRAME_COUNT)}")
    except:
        pass

    # Initialize variables for playback control
    paused = False
    frame_count = 0
    start_time = time.time()

    while True:
        if not paused:
            # grab the current frame
            frame = vs.read()
            frame = frame[1] if args.get("video", False) else frame

            # if the current frame is the last one
            if frame is None:
                break

            # resize the frame, blur it, and convert it to the hsv color space
            preprocess_frame()

            # Get current HSV values from trackbars
            # blue_lower, blue_upper = get_trackbar_values()

            # construct a mask for the color "blue", then perform a series of dilations and erosions
            mask = mask_frame()

            # find contours in the mask and initialize the current (x, y) center of the ball
            cnts = find_countours(mask)

            find_min_enclosing_circle(cnts)

            draw_locus()

            # Create a canvas to display multiple frames
            canvas_height = max(hsv_frame.shape[0], frame.shape[0], masked_frame.shape[0])
            canvas_width = hsv_frame.shape[1] + frame.shape[1] + masked_frame.shape[1]
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")

            # Place each frame in the canvas
            canvas[0:frame.shape[0], 0:frame.shape[1]] = frame
            canvas[0:hsv_frame.shape[0], frame.shape[1]:frame.shape[1]+hsv_frame.shape[1]] = hsv_frame
            canvas[0:masked_frame.shape[0], frame.shape[1] + hsv_frame.shape[1]:] = masked_frame
            # Display the canvas
            cv2.imshow("Frames", canvas)

            frame_count += 1

            if frame_count % 60 == 0:  # Print FPS every 10 frames
                fps = measure_fps(frame_count, start_time)
                print(f"FPS: {fps:.2f}")

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused

    if not args.get("video", False):
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    print(f"Main function ended in {elapsed_time}")


# def get_trackbar_values():
#     h_lower = cv2.getTrackbarPos('H Lower', 'Control Panel')
#     s_lower = cv2.getTrackbarPos('S Lower', 'Control Panel')
#     v_lower = cv2.getTrackbarPos('V Lower', 'Control Panel')
#     h_upper = cv2.getTrackbarPos('H Upper', 'Control Panel')
#     s_upper = cv2.getTrackbarPos('S Upper', 'Control Panel')
#     v_upper = cv2.getTrackbarPos('V Upper', 'Control Panel')
#     return (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper)
#
def preprocess_frame():
    global frame, hsv_frame

    frame = imutils.resize(frame, width=600)
    blured_frame = cv2.GaussianBlur(frame, (17, 17), 0)
    hsv_frame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2HSV)

def mask_frame():
    global hsv_frame, blue_lower, blue_upper, masked_frame

    mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    masked_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    return mask

def find_countours(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def find_min_enclosing_circle(cnts):
    global frame, pts
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 3:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

def draw_locus():
    global args, frame, pts

    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

# Function to measure FPS
def measure_fps(frame_count, start_time):
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    return fps

if __name__ == "__main__":
    # Profile the main function
    cProfile.run('main()', 'profile_stats')
