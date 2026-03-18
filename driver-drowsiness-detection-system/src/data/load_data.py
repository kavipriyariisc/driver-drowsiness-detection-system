def load_data(data_path):
    import pandas as pd
    import os

    # Check if the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # Load the dataset
    data_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not data_files:
        raise ValueError("No CSV files found in the specified directory.")

    data_list = []
    for file in data_files:
        file_path = os.path.join(data_path, file)
        data = pd.read_csv(file_path)
        data_list.append(data)

    # Concatenate all data into a single DataFrame
    full_data = pd.concat(data_list, ignore_index=True)
    return full_data