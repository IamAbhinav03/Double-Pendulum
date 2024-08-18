import time
from video_processing import VideoCaptureThread, FrameProcessorThread

class SeedGenerator:
    def __init__(self, bit_length: int) -> None:
        self._bit_length = bit_length
        self._captured_bits = ''
        self._processor_thread = None
        self._capture_thread = None

    def request_seed(self):
        while len(self._captured_bits) < self._bit_length:
            if not self._processor_thread.bits_queue.empty():
                self._captured_bits += self._processor_thread.bits_queue.get()
            if self._processor_thread.get_exception() or self._capture_thread.get_exception():
                raise RuntimeError("An error occurred in one of the threads")
        return self._captured_bits[:self._bit_length]
    
    def start_processing(self, source):
        self._capture_thread = VideoCaptureThread(source)
        self._processor_thread = FrameProcessorThread(self._capture_thread._frames_queue)

        print(f"Starting capture thread")
        self._capture_thread.start()
        print(f"Starting frame processing thread")
        self._processor_thread.start()

    def stop_processing(self):
        if self._capture_thread:
            self._capture_thread.stop()
            self._capture_thread.join()

        if self._processor_thread:
            self._processor_thread.stop()
            self._processor_thread.join()

        if self._capture_thread.get_exception():
            raise self._capture_thread.get_exception()

        if self._processor_thread.get_exception():
            raise self._processor_thread.get_exception()