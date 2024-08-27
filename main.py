import argparse
import time
import cv2
import os
import sys
import logging
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from random_number_generator_tarush.pendulum_tracker import PendulumTracker
from random_number_generator_tarush.rsa_number_generator import get_rsa_rabin_generator
from random_number_generator_tarush.config import BLUE_LOWER, BLUE_UPPER, RANDOM_NUMBERS_PER_FRAME

def setup_logging():
    logging.basicConfig(filename='pendulum_output.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def process_new_coordinates(tracker, generator, coords):
    current_seed = tracker.get_current_seed()
    if current_seed is not None and current_seed != process_new_coordinates.last_seed:
        generator = get_rsa_rabin_generator(current_seed)
        print(f"New seed generated: {current_seed}")
        logging.info(f"New seed generated: {current_seed}")
        logging.info(f"New coordinates: {tracker.get_pendulum_points()}")
        process_new_coordinates.last_seed = current_seed

    if generator:
        generate_random_numbers(generator)

    return generator

def generate_random_numbers(generator):
    for _ in range(RANDOM_NUMBERS_PER_FRAME):
        random_number = next(generator)
        print(f"Random number: {random_number}")
        logging.info(f"Random number: {random_number}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())

    tracker = PendulumTracker(args, BLUE_LOWER, BLUE_UPPER)
    setup_logging()
    
    generator = None
    last_coords = None
    process_new_coordinates.last_seed = None

    try:
        while True:
            frame, coords = tracker.process_frame()
            if frame is None:
                break

            if coords and coords != last_coords:
                generator = process_new_coordinates(tracker, generator, coords)
                last_coords = coords

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        tracker.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()