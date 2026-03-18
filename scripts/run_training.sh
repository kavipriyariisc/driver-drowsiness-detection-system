#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set variables
DATASET_DIR="./datasets/processed"
CHECKPOINT_DIR="./models/checkpoints"
LOG_FILE="./results/logs/training.log"

# Run the training script
python src/models/train.py --data_dir $DATASET_DIR --checkpoint_dir $CHECKPOINT_DIR | tee $LOG_FILE

# Deactivate the virtual environment
deactivate