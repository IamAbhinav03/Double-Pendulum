import unittest
import sys
import os

if __name__ == '__main__':
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # Create a test suite
    test_suite = unittest.TestSuite()

    # Discover and add tests from files matching *_tests.py
    test_loader = unittest.TestLoader()
    discovered_tests = test_loader.discover('tests', pattern='*_tests.py')
    test_suite.addTest(discovered_tests)

    # Add tests from @unit_tests.py if it exists
    unit_tests_path = os.path.join('tests', '@unit_tests.py')
    if os.path.exists(unit_tests_path):
        unit_tests = test_loader.discover('tests', pattern='@unit_tests.py')
        test_suite.addTest(unit_tests)

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(test_suite)