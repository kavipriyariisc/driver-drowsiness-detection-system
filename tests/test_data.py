# Contents of /Driver-Drowsiness-Detection/Driver-Drowsiness-Detection/tests/test_data.py

import unittest
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data

class TestDataLoading(unittest.TestCase):

    def test_load_dataset(self):
        # Test loading of dataset
        dataset = load_dataset('datasets/raw')
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)

class TestDataPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Test preprocessing of data
        raw_data = load_dataset('datasets/raw')
        processed_data = preprocess_data(raw_data)
        self.assertIsNotNone(processed_data)
        self.assertGreater(len(processed_data), 0)

if __name__ == '__main__':
    unittest.main()