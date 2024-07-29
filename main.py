import os
import random
import numpy as np
import tensorflow as tf
from config import Config
from modular_training_system import ModularTrainingSystem
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
    
    system = ModularTrainingSystem()
    
    # Train specific configurations
    system.train_single_configuration('he_normal', 'relu')
    
    
    # Load all results and analyze
    all_results = ModularTrainingSystem.load_results()
    
    performance_table = Visualizer.compare_performance(all_results)
    print("Performance Comparison:")
    print(performance_table)
    
    Visualizer.check_generalization(all_results, performance_table)