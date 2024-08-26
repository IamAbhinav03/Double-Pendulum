import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.seed_generation import (
    float_to_fixed_point,
    interleave_bits,
    hash_interleaved_bits,
    generate_seed)
from config import PENDULUM_POINTS_BUFFER
from src.rsa_number_generator import rsa_rabin_generator 

class TestRandomGenerator(unittest.TestCase):
    
    def setUp(self):
        self.sample_coords = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) 
            for _ in range(PENDULUM_POINTS_BUFFER)
        ]
        self.seed = generate_seed(self.sample_coords)
        self.rng = rsa_rabin_generator(self.seed)
        self.random_numbers = [next(self.rng) for _ in range(100000)]

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

    def test_interleave_bits(self):
        result = interleave_bits(self.sample_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_hash_interleaved_bits(self):
        interleaved = np.random.randint(2**64)
        result = hash_interleaved_bits(interleaved)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

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
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) 
            for _ in range(2)
        ]
        result = generate_seed(excess_coords)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_interleave_bits_empty_input(self):
        with self.assertRaises(ValueError):
            interleave_bits([])

    def test_hash_interleaved_bits_consistency(self):
        interleaved = np.random.randint(2**64)
        result1 = hash_interleaved_bits(interleaved)
        result2 = hash_interleaved_bits(interleaved)
        self.assertEqual(result1, result2)

    def test_generate_seed_consistency(self):
        result1 = generate_seed(self.sample_coords)
        result2 = generate_seed(self.sample_coords)
        self.assertEqual(result1, result2)

    def test_uniformity(self):
        _, p_value = chisquare(np.histogram(self.random_numbers, bins=10)[0])
        self.assertGreater(p_value, 0.05)

    def test_serial_correlation(self):
        correlation = np.corrcoef(self.random_numbers[:-1], self.random_numbers[1:])[0, 1]
        self.assertLess(abs(correlation), 0.1)

    def test_runs(self):
        median = np.median(self.random_numbers)
        runs = np.diff(np.sign(np.array(self.random_numbers) - median) != 0).sum()
        expected_runs = (len(self.random_numbers) + 2) / 2
        self.assertLess(abs(runs - expected_runs), 3 * np.sqrt(len(self.random_numbers) - 1) / 2)

    def test_serial_test(self):
        def serial_test(sequence, k=2, num_bins=16):
            n = len(sequence) - k + 1
            counts = np.zeros((num_bins,) * k)

            for i in range(n):
                index = tuple(sequence[i+j] for j in range(k))
                counts[index] += 1

            return counts.flatten()

        serial_counts = serial_test(self.random_numbers)
        _, p_value = chisquare(serial_counts)
        self.assertGreater(p_value, 0.05)

    def test_rank_of_matrices(self):
        def rank_of_matrices_test(m=32, n=32, num_matrices=1000):
            def generate_matrix(m, n):
                return np.array([next(self.rng) % 2 for _ in range(m*n)]).reshape(m, n)

            ranks = []

            for _ in range(num_matrices):
                matrix = generate_matrix(m, n)
                rank = np.linalg.matrix_rank(matrix)
                ranks.append(rank)

            return ranks

        ranks = rank_of_matrices_test()
        _, p_value = kstest(ranks, 'uniform', args=(0, 32))
        self.assertGreater(p_value, 0.05)

    def test_kolmogorov_smirnov(self):
        _, p_value = kstest(self.random_numbers, 'uniform')
        self.assertGreater(p_value, 0.05)

if __name__ == '__main__':
    unittest.main()