import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class DataLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load_and_preprocess_data(self):
        data = []
        labels = []

        for folder in os.listdir(self.dataset_dir):
            if os.path.isdir(os.path.join(self.dataset_dir, folder)):
                for img in os.listdir(os.path.join(self.dataset_dir, folder)):
                    data.append(os.path.join(self.dataset_dir, folder, img))
                    labels.append(folder)

        train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2, random_state=Config.SEED)
        val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=Config.SEED)

        train_df = pd.DataFrame({'filename': train_data, 'label': train_labels})
        val_df = pd.DataFrame({'filename': val_data, 'label': val_labels})
        test_df = pd.DataFrame({'filename': test_data, 'label': test_labels})

        return train_df, val_df, test_df