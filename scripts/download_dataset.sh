#!/bin/bash

# Script to download the driver drowsiness detection dataset

# Define the URL for the dataset
DATASET_URL="http://example.com/path/to/dataset.zip"

# Define the output directory
OUTPUT_DIR="../datasets/raw"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the dataset
echo "Downloading dataset from $DATASET_URL..."
curl -L $DATASET_URL -o $OUTPUT_DIR/dataset.zip

# Unzip the dataset
echo "Unzipping dataset..."
unzip $OUTPUT_DIR/dataset.zip -d $OUTPUT_DIR

# Remove the zip file after extraction
rm $OUTPUT_DIR/dataset.zip

echo "Dataset downloaded and extracted to $OUTPUT_DIR."