def train_model(processed_data):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # Define the model architecture
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(processed_data.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification for drowsiness detection

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(processed_data['features'], processed_data['labels'], 
                        validation_split=0.2, epochs=50, batch_size=32, 
                        callbacks=[early_stopping])

    return model, history