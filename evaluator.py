import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model):
        self.model = model

    def calculate_metrics(self, test_generator):
        predictions = self.model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        return accuracy, precision, recall, f1