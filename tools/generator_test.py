import cv2 as cv
import time

def handle_cap(cap) -> None:
    if not cap.isOpened():
        print("Error: Could not open the file. ")
        exit(0)


def fps_test1(src: str) -> None:
    print("Running fps_test1")

    cap = cv.VideoCapture(src)
    handle_cap(cap)
    
    num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print(f"Number of frames in the file: {num_frames}")
    print(f"FPS given by cv {cap.get(cv.CAP_PROP_FPS)}")
    
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of Frame");
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    
    print(f"Calculated fps: {fps}")
    cap.release()


def fps_test2(src: str) -> None:
    print("Running fps_test2")

    cap = cv.VideoCapture(src)
    handle_cap(cap)

    num_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of frame")
            break

        num_frames += 1

        # Calculate fps every 30 frames
        if num_frames % 30 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = num_frames / elapsed_time
            print(f"Live FPS: {fps:.2f}")

            # Reset counters
            start_time = time.time()
            num_frames = 0

    cap.release()


def fps_test3(src: str) -> None:
    print("Running fps_test3")

    def video_stream_generator():
        cap = cv.VideoCapture(src)
        handle_cap(cap)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of Frame")
                cap.release()
                break
            yield frame


    gen = video_stream_generator()
    num_frames = 0
    start_time = time.time()
    for frame in gen:
        num_frames += 1

    end_time = time.time()
    print(f"No. frames = {num_frames}")
    print(f"FPS: {num_frames / (end_time - start_time)}")


def main():
    print("test")
    VIDEO_FILE = "fps_test.mp4"
    fps_test1(src=VIDEO_FILE)
    fps_test2(src=VIDEO_FILE)
    fps_test3(src=VIDEO_FILE)

if __name__ == "__main__":
    main()
