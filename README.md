## Optimization of CNN through Activation Function and Kernel Initialization for Hijaiyah Letters Recognition  

### Project Overview  
This project aims to optimize a Convolutional Neural Network (CNN) by experimenting with various activation functions and kernel initializers to improve the recognition of Hijaiyah letters. By testing different combinations, this project provides insights into how these configurations affect model performance.  

### Key Features  
- **Flexible Configuration**: Set essential parameters such as dataset directory, image dimensions, batch size, number of epochs, as well as available kernel initializers and activation functions through `config.py`.
- **Data Loading and Preprocessing**: Load the dataset, perform preprocessing, and split the data into training, validation, and test sets.
- **Data Augmentation**: Utilize Keras's `ImageDataGenerator` for data augmentation.
- **Modular CNN Architecture**: Build a CNN model using modular convolutional and dense blocks, which can be optimized by experimenting with different configurations.
- **Training and Evaluation**: Train the model while saving the best-performing model and training history. Evaluate the model using metrics such as accuracy, precision, recall, and F1-score, along with confusion matrix analysis.
- **Visualization Tools**: Plot convergence curves (loss & accuracy), confusion matrices, and analyze misclassifications to better understand the model performance.

### Repository Structure  
```
├── config.py                     # Main configuration file
├── data_loader.py                # Module to load and preprocess the dataset
├── data_generator.py             # Module to create data generators using ImageDataGenerator
├── model_builder.py              # Module to build the CNN model architecture
├── model_trainer.py              # Module to train the model and save training history
├── evaluator.py                  # Module for evaluating the model (metrics calculation and confusion matrix)
├── model_training_pipeline.py    # Pipeline for overall model training and evaluation
├── visualizer.py                 # Module for visualizing training and evaluation results
├── main.py                       # Main script to run training, evaluation, and visualization
```

### Requirements  
- **Python**: Version 3.7 or higher
- **TensorFlow & Keras**: For building and training the CNN model
- **OpenCV**: For image processing
- **Pandas & NumPy**: For data manipulation
- **scikit-learn**: For computing evaluation metrics and the confusion matrix
- **Matplotlib & Seaborn**: For visualizing training and evaluation results

### Installation  
**1. Clone the Repository:**
   ```
   git clone https://github.com/khairuddin-n/Hijaiyah-Handwritten-Images-Recognition.git
   cd Hijaiyah-Handwritten-Images-Recognition
   ```
**2. Create a Virtual Environment (Optional but Recommended):**
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
**3. Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

### Dataset Structure  
The dataset used in this project is the [HMBD dataset](https://github.com/HossamBalaha/HMBD-v1/tree/master). Ensure you download the dataset from the provided link.

Once downloaded, make sure the folder names and structure match those specified in the configuration file `config.py`. The project expects the dataset to be organized in the following way:  
```
Dataset/
├── letter1/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── letter2/
│   ├── img1.png
│   └── ...
└── ...
```

### Configuration  
The `config.py` file manages various key parameters, including:
- **Directories**: Locations for the dataset (`DATASET_DIR`), models (`MODEL_DIR`), and training history (`HISTORY_DIR`).
- **Seed**: For reproducibility.
- **Training Parameters**: Such as `BATCH_SIZE`, `IMG_HEIGHT`, `IMG_WIDTH`, and `EPOCHS`.
- **Kernel Initializers & Activation Functions**: A collection of kernel initializers and activation functions to be tested.

### Running Training and Evaluation  
1. **Configure Training**:
   In `main.py`, select the desired configuration by calling the `train_single_configuration` method. For example:
   ```
   # system.train_single_configuration('he_normal', 'relu')
   # system.train_single_configuration('he_normal', 'leaky_relu')
   ```  
   Uncomment the configuration you wish to run.
2. **Run the Main Script**:
   ```
   python main.py
   ```
   This will:
   - Set the random seeds for reproducibility.
   - Load and preprocess the dataset, and create data generators.
   - Train the model using the selected configuration, saving the best model and training history.
   - Evaluate the model on the test set and save the evaluation results as a JSON file.
   - Visualize results including convergence plots and confusion matrices.

### Visualization and Analysis  
The `visualizer.py` module provides functions to:
- **Plot Convergence Curves**: Display and save loss and accuracy graphs for each training configuration.
- **Compare Performance**: Create a performance table based on evaluation metrics.
- **Plot Confusion Matrix**: Generate and save confusion matrix plots for the top and bottom 20 classes based on accuracy.
- **Analyze Misclassifications**: Analyze and save details of misclassification errors into an Excel file.

Visualization outputs are saved in:  
- `plots/` – for loss and accuracy graphs.
- `confusion_matrix/` – for confusion matrix plots.
- An Excel file for misclassification analysis will be saved according to the specified path.

### Training and Evaluation Results  
- **Best Model**: Saved in the `Models/` directory with filenames indicating the configuration (e.g., `he_normal_relu.keras`).
- **Training History**: Saved in the `histories/` directory in `.npy` and `.json` formats for further analysis.  
   
