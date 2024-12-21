import tensorflow as tf
from tensorflow.keras import layers

# Generator Model
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Encoder
    x = layers.Conv2D(64, (7, 7), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling
    for filters in [128, 256]:
        x = layers.Conv2D(filters, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(9):
        res = x
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, res])

    # Upsampling
    for filters in [128, 64]:
        x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(3, (7, 7), padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs, outputs)

generator = build_generator()
generator.summary()

# Discriminator Model
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))

    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)

    for filters in [128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(1, (4, 4), padding='same')(x)

    return tf.keras.Model(inputs, x)

discriminator = build_discriminator()
discriminator.summary()
