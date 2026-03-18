def save_model(model, file_path):
    from tensorflow.keras.models import save_model as keras_save_model
    keras_save_model(model, file_path)