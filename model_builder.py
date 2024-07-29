from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, kernel_initializer, activation):
        inputs = Input(shape=self.input_shape)
        
        def apply_activation(x):
            if activation == 'leaky_relu':
                return LeakyReLU(alpha=0.01)(x)
            else:
                return activation(x)
        
        x = self._conv_block(inputs, 16, kernel_initializer, apply_activation)
        x = self._conv_block(x, 32, kernel_initializer, apply_activation)
        x = MaxPooling2D((2, 2))(x)
        x = self._conv_block(x, 64, kernel_initializer, apply_activation)
        x = self._conv_block(x, 128, kernel_initializer, apply_activation)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = self._dense_block(x, 256, kernel_initializer, apply_activation)
        x = self._dense_block(x, 128, kernel_initializer, apply_activation)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

        return model

    def _conv_block(self, x, filters, kernel_initializer, activation):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
        x = activation(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x)
        x = activation(x)
        x = BatchNormalization()(x)
        return x

    def _dense_block(self, x, units, kernel_initializer, activation):
        x = Dense(units, kernel_initializer=kernel_initializer)(x)
        x = activation(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        return x