import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
import numpy as np

def create_cnn(input_shape):
    cnn_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return cnn_input, x

image_shape = (128, 128, 3)
dimension_shape = (2,)

vehicle_input, vehicle_features = create_cnn(image_shape)
space_input, space_features = create_cnn(image_shape)

dimensions_input = Input(shape=dimension_shape)
dimensions_features = Dense(32, activation='relu')(dimensions_input)

combined = concatenate([vehicle_features, space_features, dimensions_features])
x = Dense(64, activation='relu')(combined)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[vehicle_input, space_input, dimensions_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

vehicle_images = np.random.rand(100, 128, 128, 3)
space_images = np.random.rand(100, 128, 128, 3)
dimensions = np.random.rand(100, 2)
labels = np.random.randint(0, 2, 100)

model.fit([vehicle_images, space_images, dimensions], labels, epochs=10, batch_size=8)
