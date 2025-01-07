import os
import json
from config import Config
from data_loader import DataLoader
from data_generator import DataGenerator
from model_builder import ModelBuilder
from model_trainer import ModelTrainer
from evaluator import Evaluator

class ModelTrainingPipeline:
    def __init__(self):
        self.data_loader = DataLoader(Config.DATASET_DIR)
        self.data_generator = DataGenerator(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.BATCH_SIZE)
        self.train_df, self.val_df, self.test_df = self.data_loader.load_and_preprocess_data()
        self.train_generator, self.validation_generator, self.test_generator = self.data_generator.create_data_generators(
            self.train_df, self.val_df, self.test_df, shuffle_seed=Config.SEED
        )
        self.num_classes = len(self.train_generator.class_indices)
        
    def train_single_configuration(self, initializer_name, activation_name):
        print(f'Training and evaluating: Initializer={initializer_name}, Activation={activation_name}')
        
        kernel_initializer = Config.KERNEL_INITIALIZERS[initializer_name]
        activation = Config.ACTIVATION_FUNCTIONS[activation_name]
        
        model_builder = ModelBuilder(input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), num_classes=self.num_classes)
        model = model_builder.build_model(kernel_initializer, activation)
        
        trainer = ModelTrainer(model, initializer_name, activation_name, Config.MODEL_DIR, Config.HISTORY_DIR)
        history = trainer.train_model(self.train_generator, self.validation_generator, epochs=Config.EPOCHS)
        
        evaluator = Evaluator(model=model)
        accuracy, precision, recall, f1 = evaluator.calculate_metrics(self.test_generator)
        
        results = {
            'initializer': initializer_name,
            'activation': activation_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'history': history.history
        }
        
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        filename = f"{results['initializer']}_{results['activation']}_results.json"
        filepath = os.path.join(Config.HISTORY_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f)
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load_results():
        results = []
        for filename in os.listdir(Config.HISTORY_DIR):
            if filename.endswith('_results.json'):
                filepath = os.path.join(Config.HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
        return results
