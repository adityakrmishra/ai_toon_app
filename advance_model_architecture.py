import tensorflow as tf
from tensorflow.keras import layers

# Define a U-Net generator model
def build_unet_generator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Encoder
    down1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    down1 = layers.LeakyReLU(0.2)(down1)

    down2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU(0.2)(down2)

    down3 = layers.Conv2D(256, (4, 4), strides=2, padding='same')(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU(0.2)(down3)

    down4 = layers.Conv2D(512, (4, 4), strides=2, padding='same')(down3)
    down4 = layers.BatchNormalization()(down4)
    down4 = layers.LeakyReLU(0.2)(down4)

    # Bottleneck
    bottleneck = layers.Conv2D(1024, (4, 4), strides=2, padding='same')(down4)
    bottleneck = layers.ReLU()(bottleneck)

    # Decoder
    up4 = layers.Conv2DTranspose(512, (4, 4), strides=2, padding='same')(bottleneck)
    up4 = layers.BatchNormalization()(up4)
    up4 = layers.ReLU()(up4)
    up4 = layers.Concatenate()([up4, down4])

    up3 = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same')(up4)
    up3 = layers.BatchNormalization()(up3)
    up3 = layers.ReLU()(up3)
    up3 = layers.Concatenate()([up3, down3])

    up2 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(up3)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.ReLU()(up2)
    up2 = layers.Concatenate()([up2, down2])

    up1 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(up2)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.ReLU()(up1)
    up1 = layers.Concatenate()([up1, down1])

    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')(up1)

    return tf.keras.Model(inputs, outputs)

unet_generator = build_unet_generator()
unet_generator.summary()

# Define a ResNet block
def resnet_block(input_layer, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, input_layer])
    return x

# Define a ResNet generator model
def build_resnet_generator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Initial Convolutional Layer
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
        x = resnet_block(x, 256)

    # Upsampling
    for filters in [128, 64]:
        x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(3, (7, 7), padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs, outputs)

resnet_generator = build_resnet_generator()
resnet_generator.summary()
