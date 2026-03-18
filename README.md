# Driver Drowsiness Detection

This project implements a Driver Drowsiness Detection system using deep learning techniques. The goal is to monitor the driver's state and detect drowsiness to enhance road safety.

## Project Structure

```
Driver-Drowsiness-Detection
├── datasets
│   ├── raw                # Raw dataset files for training and testing
│   ├── processed          # Processed dataset files ready for model training
│   └── annotations        # Annotation files including labels and bounding boxes
├── models
│   ├── checkpoints        # Model checkpoints during training
│   ├── exports            # Exported models for inference
│   └── model_definition.py # Defines the architecture of the model
├── src
│   ├── __init__.py       # Marks the src directory as a Python package
│   ├── data
│   │   ├── load_data.py   # Functions to load the dataset
│   │   └── preprocess.py   # Functions for data preprocessing
│   ├── models
│   │   ├── architecture.py  # Model architecture definition
│   │   └── train.py         # Training loop and logic
│   ├── inference
│   │   └── predict.py       # Functions for making predictions
│   ├── utils
│   │   └── helpers.py       # Utility functions
│   └── visualization
│       └── vis.py          # Functions for visualizing results
├── notebooks
│   ├── 00-exploratory-data-analysis.ipynb # Exploratory data analysis
│   └── 01-training.ipynb                  # Training process and results
├── results
│   ├── experiments
│   │   └── exp_01         # Results of the first experiment
│   ├── logs
│   │   └── training.log    # Logs of the training process
│   └── reports
│       └── evaluation.md   # Model performance evaluation report
├── tests
│   ├── test_data.py       # Unit tests for data functions
│   └── test_models.py     # Unit tests for model functions
├── scripts
│   ├── download_dataset.sh  # Script to download the dataset
│   └── run_training.sh      # Script to run the training process
├── requirements.txt        # Project dependencies
├── .gitignore              # Files to ignore in version control
├── LICENSE                 # Licensing information
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Driver-Drowsiness-Detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset by placing raw files in the `datasets/raw` directory.
2. Run the preprocessing script to prepare the data:
   ```
   python src/data/preprocess.py
   ```
3. Train the model using the training script:
   ```
   bash scripts/run_training.sh
   ```
4. Evaluate the model and visualize results using the provided notebooks.

## License

This project is licensed under the MIT License. See the LICENSE file for details.