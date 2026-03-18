import unittest
from src.models.architecture import ModelArchitecture
from src.models.train import train_model

class TestModelArchitecture(unittest.TestCase):
    def test_model_initialization(self):
        model = ModelArchitecture()
        self.assertIsNotNone(model)

    def test_model_output_shape(self):
        model = ModelArchitecture()
        input_shape = (1, 224, 224, 3)  # Example input shape
        output = model(input_shape)
        self.assertEqual(output.shape, (1, 10))  # Assuming 10 classes for output

class TestModelTraining(unittest.TestCase):
    def test_training_process(self):
        model = ModelArchitecture()
        dataset = "path/to/dataset"  # Replace with actual dataset path
        epochs = 5
        history = train_model(model, dataset, epochs)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

if __name__ == '__main__':
    unittest.main()