import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(raw_data_dir):
    data_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    data = []
    for file in data_files:
        file_path = os.path.join(raw_data_dir, file)
        df = pd.read_csv(file_path)
        data.append(df)
    return pd.concat(data, ignore_index=True)

def load_processed_data(processed_data_dir):
    data_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
    data = []
    for file in data_files:
        file_path = os.path.join(processed_data_dir, file)
        df = pd.read_csv(file_path)
        data.append(df)
    return pd.concat(data, ignore_index=True)

def load_annotations(annotations_dir):
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    annotations = []
    for file in annotation_files:
        file_path = os.path.join(annotations_dir, file)
        with open(file_path, 'r') as f:
            annotations.append(json.load(f))
    return annotations

def split_data(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)