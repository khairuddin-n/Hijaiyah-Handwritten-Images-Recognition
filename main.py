import os
import random
import numpy as np
import tensorflow as tf
from config import Config
from model_training_pipeline import ModelTrainingPipeline
from tensorflow.keras.models import load_model
from visualizer import Visualizer
import warnings

warnings.filterwarnings("ignore")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")

if __name__ == "__main__":
    set_seeds(Config.SEED)

    system = ModelTrainingPipeline()

    # Train specific configurations
    # system.train_single_configuration('he_normal', 'relu')
    # system.train_single_configuration('he_normal', 'leaky_relu')
    # ...

    # Load all results and analyze
    all_results = ModelTrainingPipeline.load_results()

    # Plot individual loss and accuracy for all configurations
    # Visualizer.plot_convergence(all_results, save_dir="plots")

    model_path = os.path.join(Config.MODEL_DIR, "he_normal_relu.keras")
    model = load_model(model_path)

    # Performance comparison
    performance_table = Visualizer.compare_performance(all_results)
    print("Performance Comparison:")
    print(performance_table)

    # Save performance table to an Excel file
    # performance_table_file = "performance_comparison.xlsx"
    # performance_table.to_excel(performance_table_file, index=False)
    # print(f"Performance table saved to {performance_table_file}")

    # Visualisasi confusion matrix
    Visualizer.plot_confusion_matrix(
        model, 
        system.test_generator
    )
