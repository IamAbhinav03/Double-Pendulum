import cv2 as cv
import time

def main():
    # Open the default webcam
    cap = cv.VideoCapture('fps_test.mp4')

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(0)

    # Set desired resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

    # Set desired FPS (this may not always work, depending on the camera)
    # cap.set(cv.CAP_PROP_FPS, 30)

    cv.namedWindow("Webcam Stream")
    
    # Variables for calculating FPS
    num_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        num_frames += 1

        # Calculate FPS every 30 frames
        if num_frames % 30 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = num_frames / elapsed_time
            print(f"FPS: {fps:.2f}")
            # Reset counters
            start_time = time.time()
            num_frames = 0

        cv.imshow("Webcam Stream", frame)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
