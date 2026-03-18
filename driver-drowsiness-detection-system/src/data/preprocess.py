def preprocess_data(data):
    # Perform necessary preprocessing steps
    # Example steps could include:
    # - Handling missing values
    # - Normalizing or scaling features
    # - Encoding categorical variables
    # - Splitting data into features and labels

    # Assuming 'data' is a pandas DataFrame
    # Here is a placeholder for preprocessing logic

    # Example: Drop rows with missing values
    data = data.dropna()

    # Example: Normalize feature columns (assuming they are numeric)
    feature_columns = data.columns[:-1]  # Assuming the last column is the label
    data[feature_columns] = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()

    # Split features and labels
    X = data[feature_columns]
    y = data.iloc[:, -1]  # Assuming the last column is the label

    return X, y