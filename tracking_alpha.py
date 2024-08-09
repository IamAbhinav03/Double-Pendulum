import cv2
import queue
from collections import deque
from tools.fps_counter import FPS
import time
from video_processing import VideoCaptureThread, FrameProcessorThread

def main():
    """
    Main function that sets up the video capture and frame processing threads and runs the main loop.
    """
    source = "sample.mp4"
    # source = 0
    blue_lower = (102, 41, 2)
    blue_upper = (179, 255, 255)
    frame_queue = queue.Queue(maxsize=5)
    processed_frame_queue = queue.Queue(maxsize=5)

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

if __name__ == "__main__":
    main()