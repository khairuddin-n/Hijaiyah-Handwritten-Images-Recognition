import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

class ModelTrainer:
    def __init__(self, model, initializer_name, activation_name, model_dir, history_dir):
        self.model = model
        self.initializer_name = initializer_name
        self.activation_name = activation_name
        self.model_dir = model_dir
        self.history_dir = history_dir

    def train_model(self, train_generator, validation_generator, epochs=20):
        model_filename = os.path.join(self.model_dir, f"{self.initializer_name}_{self.activation_name}.keras")
        checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[checkpoint])

        history_filename = os.path.join(self.history_dir, f"{self.initializer_name}_{self.activation_name}_history.npy")
        np.save(history_filename, history.history)

        return history