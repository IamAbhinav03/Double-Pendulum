import unittest
from unittest.mock import patch
import random
import hashlib
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.seed_generation import (
    float_to_fixed_point,
    interleave_bits,
    hash_interleaved_bits,
    generate_seed)

class TestRandomGenerator(unittest.TestCase):
    
    def setUp(self):
        self.sample_coords = [
            (random.uniform(-10, 10), random.uniform(-10, 10)) 
            for _ in range(5)
        ]

    def test_float_to_fixed_point(self):
        test_cases = [
            (1.5, 1500000),
            (2.75, 2750000),
            (-3.25, -3250000),
            (0, 0),
            (1000000.5, 1000000500000),
        ]
        for input_value, expected in test_cases:
            with self.subTest(input_value=input_value):
                self.assertEqual(float_to_fixed_point(input_value), expected)

    def test_float_to_fixed_point_precision(self):
        self.assertEqual(float_to_fixed_point(1.234567), 1234567)

    def test_interleave_bits(self):
        result = interleave_bits(self.sample_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_hash_interleaved_bits(self):
        interleaved = random.getrandbits(64)  # Generate a random 64-bit integer
        result = hash_interleaved_bits(interleaved)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)  # SHA-256 produces a 64-character hexadecimal string

    @patch('random.getrandbits')
    def test_hash_interleaved_bits_deterministic(self, mock_getrandbits):
        mock_getrandbits.return_value = 12345
        result = hash_interleaved_bits(12345)
        expected = hashlib.sha256(str(12345).encode()).hexdigest()
        self.assertEqual(result, expected)

    def test_generate_seed(self):
        result = generate_seed(self.sample_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_generate_seed_with_insufficient_coords(self):
        insufficient_coords = self.sample_coords[:3]
        result = generate_seed(insufficient_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_generate_seed_with_excess_coords(self):
        excess_coords = self.sample_coords + [
            (random.uniform(-10, 10), random.uniform(-10, 10)) 
            for _ in range(2)
        ]
        result = generate_seed(excess_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_interleave_bits_empty_input(self):
        with self.assertRaises(ValueError):
            interleave_bits([])

    def test_hash_interleaved_bits_consistency(self):
        interleaved = random.getrandbits(64)  # Generate a random 64-bit integer
        result1 = hash_interleaved_bits(interleaved)
        result2 = hash_interleaved_bits(interleaved)
        self.assertEqual(result1, result2)

    def test_generate_seed_consistency(self):
        result1 = generate_seed(self.sample_coords)
        result2 = generate_seed(self.sample_coords)
        self.assertEqual(result1, result2)

if __name__ == '__main__':
    unittest.main()