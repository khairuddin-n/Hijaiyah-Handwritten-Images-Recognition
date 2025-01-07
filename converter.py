import tensorflow as tf
import os

def convert_keras_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a .keras model to .tflite format and saves it to a specified directory.

    Parameters:
    keras_model_path (str): Path to the .keras model file.
    tflite_model_path (str): Path to save the converted .tflite model file.
    """
    try:
        # Load the .keras model
        model = tf.keras.models.load_model(keras_model_path)
        print(f"Model loaded successfully from {keras_model_path}")

        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print("Model successfully converted to TFLite format.")

        # Save the TFLite model to the specified path
        os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_model_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
keras_model_path = "Models/he_normal_relu.keras" 
tflite_model_path = "Models/tflite/model.tflite"  

convert_keras_to_tflite(keras_model_path, tflite_model_path)
