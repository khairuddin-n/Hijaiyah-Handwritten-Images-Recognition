import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_convergence(histories, title):
        plt.figure(figsize=(12, 8))
        for label, history in histories.items():
            plt.plot(history['val_loss'], label=f'{label} (val)')
            plt.plot(history['loss'], label=f'{label} (train)', linestyle='--')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def compare_performance(results):
        df = pd.DataFrame(results)
        return df.sort_values('accuracy', ascending=False)

    @staticmethod
    def check_generalization(results, performance_table):
        for index, row in performance_table.iterrows():
            initializer = row['initializer']
            activation = row['activation']
            history = next(r['history'] for r in results if r['initializer'] == initializer and r['activation'] == activation)
            train_acc = history['accuracy'][-1]
            val_acc = history['val_accuracy'][-1]
            test_acc = row['accuracy']
            print(f"\nGeneralization check for {initializer} - {activation}:")
            print(f"Final training accuracy: {train_acc:.4f}")
            print(f"Final validation accuracy: {val_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
            if train_acc - val_acc > 0.1 or train_acc - test_acc > 0.1:
                print("Warning: Possible overfitting detected.")
            elif val_acc - test_acc > 0.05:
                print("Warning: Model might not generalize well to unseen data.")
            else:
                print("Model seems to generalize well.")