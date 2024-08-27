import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from random_number_generator_tarush.pendulum_tracker import PendulumTracker
from random_number_generator_tarush.seed_generation import generate_seed
from random_number_generator_tarush.rsa_number_generator import get_rsa_rabin_generator
from random_number_generator_tarush.config import PENDULUM_POINTS_BUFFER, RANDOM_NUMBER_MODULUS

class IntegrationTests(unittest.TestCase):

    def setUp(self):
        self.mock_args = {'buffer': 64, 'video': None}
        self.blue_lower = (100, 50, 50)
        self.blue_upper = (130, 255, 255)

    @patch('src.pendulum_tracker.VideoStream')
    def test_pendulum_to_seed_generation(self, mock_video_stream):
        # Mock video stream
        mock_video_stream.return_value.start.return_value.read.return_value = MagicMock()

        # Initialize PendulumTracker
        tracker = PendulumTracker(self.mock_args, self.blue_lower, self.blue_upper)

        # Simulate pendulum movement
        test_points = [(100, 100), (110, 110), (120, 120), (130, 130), (140, 140)]
        for point in test_points:
            tracker.add_unique_point(point)

        # Check if seed was generated
        seed = tracker.get_current_seed()
        self.assertIsNotNone(seed)
        self.assertIsInstance(seed, int)

        # Verify pendulum points
        pendulum_points = tracker.get_pendulum_points()
        self.assertEqual(len(pendulum_points), min(len(test_points), PENDULUM_POINTS_BUFFER))
        self.assertEqual(pendulum_points, test_points[:PENDULUM_POINTS_BUFFER])

    def test_seed_to_rsa_number_generation(self):
        # Generate a seed
        test_points = [(100, 100), (110, 110), (120, 120), (130, 130), (140, 140)]
        seed = generate_seed(test_points)
        self.assertIsNotNone(seed)

        # Get RSA-Rabin generator
        rsa_gen = get_rsa_rabin_generator(seed)

        # Generate some numbers
        generated_numbers = [next(rsa_gen) for _ in range(10)]

        # Check properties of generated numbers
        for num in generated_numbers:
            self.assertIsInstance(num, int)
            self.assertGreaterEqual(num, 0)
            self.assertLess(num, RANDOM_NUMBER_MODULUS)

        # Check that generated numbers are not all the same
        self.assertGreater(len(set(generated_numbers)), 1)

    @patch('src.pendulum_tracker.VideoStream')
    def test_full_integration(self, mock_video_stream):
        # Mock video stream
        mock_video_stream.return_value.start.return_value.read.return_value = MagicMock()

        # Initialize PendulumTracker
        tracker = PendulumTracker(self.mock_args, self.blue_lower, self.blue_upper)

        # Simulate pendulum movement
        test_points = [(100, 100), (110, 110), (120, 120), (130, 130), (140, 140)]
        for point in test_points:
            tracker.add_unique_point(point)

        # Get generated seed
        seed = tracker.get_current_seed()
        self.assertIsNotNone(seed)

        # Get RSA-Rabin generator
        rsa_gen = get_rsa_rabin_generator(seed)

        # Generate some numbers
        generated_numbers = [next(rsa_gen) for _ in range(10)]

        # Check properties of generated numbers
        for num in generated_numbers:
            self.assertIsInstance(num, int)
            self.assertGreaterEqual(num, 0)
            self.assertLess(num, RANDOM_NUMBER_MODULUS)

        # Check that generated numbers are not all the same
        self.assertGreater(len(set(generated_numbers)), 1)

    def test_error_handling(self):
        # Test with insufficient points
        insufficient_points = [(100, 100), (110, 110), (120, 120)]
        seed = generate_seed(insufficient_points)
        self.assertIsNone(seed)

        # Test with invalid points
        invalid_points = [(100, 100), (110, 110), (120, 120), (130, 130), "invalid"]
        with self.assertRaises(TypeError):
            generate_seed(invalid_points)

if __name__ == '__main__':
    unittest.main()