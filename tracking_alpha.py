import argparse
import time
import cProfile
import cv2
from video_processing import VideoCaptureThread, VideoProcessor, VideoDisplayThread


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video object detection and tracking.")
    parser.add_argument("-v", "--video", help="Path to the video file.")
    parser.add_argument("-b", "--buffer", type=int, default=64, help="Max buffer size.")
    args = parser.parse_args()

    # Start video capture
    if not args.video:
        print("Webcam input is currently not supported.")
        return
    else:
        print(f"Opening video file: {args.video}")
        video_capture = VideoCaptureThread(source=args.video).start()

    # Start video processing
    video_processor = VideoProcessor(frame=video_capture.frame, buffer_size=args.buffer).start()


    frame_count = 0

    # Main loop
    while True:
        start_time = time.time()

        if video_capture.stopped or video_processor.stopped:
            video_capture.stop()
            video_processor.stop()
            break

        frame = video_processor.frame
        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            print("Exiting program.")
            break

        video_processor.frame = video_capture.frame

        frame_count += 1
        if frame_count % 60 == 0:  # Print FPS every 60 frames
            fps = measure_fps(frame_count, start_time)
            print(f"FPS: {fps:.2f}")

    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    print(f"Main function ended in {elapsed_time:.2f} seconds.")


def measure_fps(frame_count: int, start_time: float) -> float:
    """Calculate and return the frames per second."""
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    return fps


if __name__ == "__main__":
    # Profile the main function
    cProfile.run('main()', 'profile_stats')