import cv2 as cv
import numpy as np
import argparse

erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'

# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE

def erosion(val, src):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
    
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)

def dilatation(val, src):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
    
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilation_window, dilatation_dst)

def main():
    # Open the default webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(0)
    
    frame_rate = True
    if frame_rate:
        print(cap.get(cv.CAP_PROP_FPS))
        frame_rate = False

    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, lambda x: None)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, lambda x: None)
    
    cv.namedWindow(title_dilation_window)
    cv.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, lambda x: None)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, lambda x: None)

    while True:
        ret, src = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        erosion(0, src)
        dilatation(0, src)

        # Exit on pressing 'q'
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
