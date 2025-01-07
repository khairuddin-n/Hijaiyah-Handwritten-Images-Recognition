import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config
from data_loader import DataLoader
from data_generator import DataGenerator
from model_training_pipeline import ModelTrainingPipeline

def evaluate_tflite_model(tflite_model_path, test_generator):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true = []
    y_pred = []

    # Iterate over the test data
    for batch_data, batch_labels in test_generator:
        # Stop when enough data has been processed
        if len(y_true) >= test_generator.samples:
            break

        for i in range(batch_data.shape[0]):  # Process each sample individually
            input_data = np.expand_dims(batch_data[i], axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            batch_pred = np.argmax(output_data, axis=1)

            # Append predictions and true labels
            y_true.append(np.argmax(batch_labels[i]))  # Convert one-hot to integer
            y_pred.append(batch_pred[0])

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":

    tflite_model_path = "Models/tflite/model.tflite"

    # Prepare data
    data_loader = DataLoader(Config.DATASET_DIR)
    data_generator = DataGenerator(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.BATCH_SIZE)

    # Load and preprocess test data
    _, _, test_df = data_loader.load_and_preprocess_data()

    # Create test generator
    test_generator = data_generator._create_generator(
        datagen=tf.keras.preprocessing.image.ImageDataGenerator(),
        df=test_df,
        shuffle=False
    )

    # Evaluate TFLite model
    tflite_metrics = evaluate_tflite_model(tflite_model_path, test_generator)
    print("TFLite Model Performance:")
    print(f"Accuracy: {tflite_metrics['accuracy']}")
    print(f"Precision: {tflite_metrics['precision']}")
    print(f"Recall: {tflite_metrics['recall']}")
    print(f"F1 Score: {tflite_metrics['f1']}")

    # Compare with previous results
    print("\nComparison with previous configurations:")
    all_results = ModelTrainingPipeline.load_results()
    for result in all_results:
        # Filter results to only show those with 'relu' activation and 'he_normal' initializer
        if result['initializer'] == 'he_normal' and result['activation'] == 'relu':
            print(f"Initializer: {result['initializer']}, Activation: {result['activation']}")
            print(f"Accuracy: {result['accuracy']}, Precision: {result['precision']}, Recall: {result['recall']}, F1: {result['f1']}")
            print("---")

