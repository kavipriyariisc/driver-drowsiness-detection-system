# Driver Drowsiness Detection System

This project implements a driver drowsiness detection system using machine learning techniques. The goal is to monitor the driver's state and detect signs of drowsiness to enhance road safety.

## Project Structure

- **datasets/processed**: Contains the processed dataset used for training the model.
- **models/exports**: Directory for storing the exported trained models.
- **notebooks/01-training.ipynb**: Jupyter notebook for training the driver drowsiness detection model, including data loading, preprocessing, model training, and evaluation.
- **src/data/load_data.py**: Contains the `load_data` function to load the dataset from a specified path.
- **src/data/preprocess.py**: Includes the `preprocess_data` function to preprocess the loaded data for model training.
- **src/models/train.py**: Exports the `train_model` function that trains the model using the preprocessed data and returns the trained model and training history.
- **src/utils/helpers.py**: Includes the `save_model` function to save the trained model to a specified file path.
- **requirements.txt**: Lists the dependencies required for the project, which can be installed using pip.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd driver-drowsiness-detection-system
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset and place it in the `datasets/processed` directory.
2. Open the Jupyter notebook `notebooks/01-training.ipynb` to train the model.
3. Follow the instructions in the notebook to load, preprocess the data, train the model, and evaluate its performance.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.