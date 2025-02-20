import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

class Visualizer:
    @staticmethod
    def plot_convergence(results, filter_by=None, value=None, save_dir="plots"):
        """
        Plot individual histories (loss and accuracy) based on specific parameters and save plots
        in separate directories for loss and accuracy.

        Parameters:
        - results: List of dictionaries containing training results.
        - filter_by: Parameter to filter by ('initializer', 'activation', or None for all).
        - value: Value of the parameter to filter (e.g., 'he_normal' or 'relu').
        - save_dir: Base directory to save the plots.
        """
        loss_dir = os.path.join(save_dir, "loss")
        accuracy_dir = os.path.join(save_dir, "accuracy")
        os.makedirs(loss_dir, exist_ok=True)
        os.makedirs(accuracy_dir, exist_ok=True)

        filtered_results = results
        if filter_by and value:
            filtered_results = [
                result for result in results if result[filter_by] == value
            ]
            if not filtered_results:
                print(f"No results found for {filter_by} = {value}")
                return

        for result in filtered_results:
            label = f"{result['initializer']}_{result['activation']}"
            history = result['history']

            # Plot Loss
            plt.figure(figsize=(10, 6))
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.plot(history['loss'], label='Training Loss', linestyle='--')
            plt.title(f"Loss Convergence: {label}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            loss_file_path = os.path.join(loss_dir, f"{label}_loss_plot.png")
            plt.savefig(loss_file_path)
            print(f"Saved loss plot for {label} to {loss_file_path}")
            plt.close()

            # Plot Accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.plot(history['accuracy'], label='Training Accuracy', linestyle='--')
            plt.title(f"Accuracy Convergence: {label}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            acc_file_path = os.path.join(accuracy_dir, f"{label}_accuracy_plot.png")
            plt.savefig(acc_file_path)
            print(f"Saved accuracy plot for {label} to {acc_file_path}")
            plt.close()

    @staticmethod
    def compare_performance(results):
        df = pd.DataFrame(results)
        return df.sort_values('accuracy', ascending=False)

    @staticmethod
    def plot_confusion_matrix(model, test_generator, save_dir="confusion_matrix"):
        """
        Calculate and plot confusion matrix for top and bottom 20 classes based on accuracy,
        with additional context about class frequency.

        Parameters:
        - model: Trained Keras model
        - test_generator: Test data generator
        - save_dir: Directory to save confusion matrix plots
        """
        os.makedirs(save_dir, exist_ok=True)

        predictions = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        class_names = list(test_generator.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred)
        class_accuracies = np.diagonal(cm) / np.sum(cm, axis=1)
        class_frequencies = np.bincount(y_true)

        class_info = [
            {
                'class_name': class_names[i],
                'accuracy': class_accuracies[i],
                'frequency': class_frequencies[i]
            }
            for i in range(len(class_names))
        ]

        class_info_sorted = sorted(class_info, key=lambda x: x['accuracy'], reverse=True)
        top_class_info = class_info_sorted[:20]
        bottom_class_info = class_info_sorted[-20:]

        def plot_cm(cm_subset, class_names_subset, title, filename):
            cm_normalized = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(15, 12))
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu',
                        xticklabels=class_names_subset, yticklabels=class_names_subset)
            plt.title(title)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            save_path = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        top_cm = cm[[class_names.index(item['class_name']) for item in top_class_info]]
        top_cm = top_cm[:, [class_names.index(item['class_name']) for item in top_class_info]]
        plot_cm(top_cm, [item['class_name'] for item in top_class_info],
                'Confusion Matrix (Top 20 Classes)', 'top_20_classes.png')

        bottom_cm = cm[[class_names.index(item['class_name']) for item in bottom_class_info]]
        bottom_cm = bottom_cm[:, [class_names.index(item['class_name']) for item in bottom_class_info]]
        plot_cm(bottom_cm, [item['class_name'] for item in bottom_class_info],
                'Confusion Matrix (Bottom 20 Classes)', 'bottom_20_classes.png')

    @staticmethod
    def analyze_misclassifications(model, test_generator, save_path="misclassifications_analysis.xlsx"):
        """
        Analyze misclassifications and save details of class-to-class errors.

        Parameters:
        - model: Trained Keras model
        - test_generator: Test data generator
        - save_path: File path to save the misclassifications analysis table
        """
        predictions = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        class_names = list(test_generator.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred)

        misclassification_data = [
            {
                "True Class": class_names[i],
                "Predicted Class": class_names[j],
                "Count": cm[i, j]
            }
            for i in range(len(class_names))
            for j in range(len(class_names))
            if i != j and cm[i, j] > 0
        ]

        df_misclassifications = pd.DataFrame(misclassification_data)
        df_misclassifications = df_misclassifications.sort_values(by="Count", ascending=False)

        df_misclassifications.to_excel(save_path, index=False)
        print(f"Misclassifications analysis saved to {save_path}")
