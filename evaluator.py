import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model):
        self.model = model

    def calculate_metrics(self, test_generator):
        predictions = self.model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        return accuracy, precision, recall, f1
    
    def calculate_confusion_matrix(self, test_generator):
        """
        Calculate confusion matrix for the test data.
        
        Parameters:
        - test_generator: Data generator for test data
        
        Returns:
        - confusion_matrix: Numpy array of confusion matrix
        - class_names: List of class names
        """
        predictions = self.model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        # Get class names
        class_names = list(test_generator.class_indices.keys())
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return cm, class_names

    def calculate_per_class_metrics(self, test_generator):
        """
        Calculate metrics for each class.
        
        Parameters:
        - test_generator: Data generator for test data
        
        Returns:
        - per_class_metrics: Dictionary with metrics for each class
        """
        predictions = self.model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        # Get class names
        class_names = list(test_generator.class_indices.keys())
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for class_index, class_name in enumerate(class_names):
            # Create binary classification for this class
            y_true_binary = (y_true == class_index).astype(int)
            y_pred_binary = (y_pred == class_index).astype(int)
            
            per_class_metrics[class_name] = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary),
                'recall': recall_score(y_true_binary, y_pred_binary),
                'f1_score': f1_score(y_true_binary, y_pred_binary)
            }
        
        return per_class_metrics
