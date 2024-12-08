import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    def __init__(self, img_height, img_width, batch_size):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def create_data_generators(self, train_df, val_df, test_df, shuffle_seed=None):
        train_datagen = ImageDataGenerator()
        val_test_datagen = ImageDataGenerator()

        train_generator = self._create_generator(train_datagen, train_df, shuffle=True, seed=shuffle_seed)
        validation_generator = self._create_generator(val_test_datagen, val_df, shuffle=False, seed=shuffle_seed)
        test_generator = self._create_generator(val_test_datagen, test_df, shuffle=False)

        return train_generator, validation_generator, test_generator

    def _create_generator(self, datagen, df, shuffle, seed=None):
        return datagen.flow_from_dataframe(
            dataframe=df,
            x_col='filename',
            y_col='label',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=shuffle,
            seed=seed,
            preprocessing_function=self.preprocess_image)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_height, self.img_width))
        img = cv2.bitwise_not(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
