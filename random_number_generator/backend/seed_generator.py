import time
from threading import Lock
from video_processing import VideoCaptureThread, FrameProcessorThread

# A lock to ensure that only one thread can modify shared resources at a time
lock = Lock()

class SeedGenerator:
    def __init__(self, bit_length: int) -> None:
        """
        Initializes the SeedGenerator with the desired bit length.
        """
        self._bit_length: int = bit_length  # Number of random bits needed
        self._captured_bits: str = ''  # Stores the captured bits as a string
        self._processor_thread: Optional[FrameProcessorThread] = None  # Thread that processes video frames
        self._capture_thread: Optional[VideoCaptureThread] = None  # Thread that captures video frames

    def request_seed(self) -> str:
        """
        Continuously collects bits from the processor thread until we have enough to meet the requested bit length.
        Returns the collected bits as a seed.
        """
        while len(self._captured_bits) < self._bit_length:  # Keep collecting bits until we have enough
            if not self._processor_thread.bits_queue.empty():  # Check if there are any bits in the queue
                self._captured_bits += self._processor_thread.bits_queue.get()  # Add bits to the captured bits

            # Check if any thread encountered an exception
            if self._processor_thread.get_exception() or self._capture_thread.get_exception():
                raise RuntimeError("An error occurred in one of the threads")

        return self._captured_bits[:self._bit_length]  # Return only the number of bits requested

    def start_processing(self, source: str) -> None:
        """
        Starts the video capture and frame processing threads.
        """
        self._capture_thread = VideoCaptureThread(source)  # Create the capture thread with the video source
        self._processor_thread = FrameProcessorThread(self._capture_thread._frames_queue)  # Create the processor thread

        print("Starting capture thread")
        self._capture_thread.start()  # Start the capture thread
        print("Starting frame processing thread")
        self._processor_thread.start()  # Start the frame processing thread

    def stop_processing(self) -> None:
        """
        Stops both the video capture and frame processing threads.
        Raises any exceptions that occurred in the threads.
        """
        print("Stopping process")

        if self._processor_thread:
            print("Stopping processor thread")
            self._processor_thread.stop()  # Stop the processor thread

            
        if self._capture_thread:
            print("Stopping capture thread")
            self._capture_thread.stop()  # Stop the capture thread

        

        # If there was an exception in the capture thread, raise it
        if self._capture_thread.get_exception():
            raise self._capture_thread.get_exception()

        # If there was an exception in the processor thread, raise it
        if self._processor_thread.get_exception():
            raise self._processor_thread.get_exception()