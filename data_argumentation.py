import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define an ImageDataGenerator with advanced augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=50.0,
    rescale=1./255
)

# Load an example image
image_path = 'path_to_image.jpg'
image = tf.keras.preprocessing.image.load_img(image_path)
x = tf.keras.preprocessing.image.img_to_array(image)
x = x.reshape((1,) + x.shape)

# Generate augmented images and save them to a directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # Generate 20 augmented images

# Using tf.data for more control over augmentation
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
    return image

# Apply augmentation to a dataset
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = augment_image(image)
    return image

# Example dataset
image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess_image)

# Visualize augmented images
import matplotlib.pyplot as plt

for image in dataset.take(5):
    plt.figure()
    plt.imshow(image.numpy().astype("uint8"))
    plt.axis('off')
    plt.show()
