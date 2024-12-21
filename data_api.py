import tensorflow as tf

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.per_image_standardization(image)
    return image, label

def load_dataset(filenames, labels, batch_size=32, buffer_size=1000, cache=True, prefetch=True, parallel_calls=tf.data.experimental.AUTOTUNE):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=parallel_calls)
    
    if cache:
        dataset = dataset.cache()  # Cache the dataset in memory to improve performance
    
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch data to overlap data preprocessing and model execution
    
    return dataset

# Example usage
filenames = ['path_to_image1.jpg', 'path_to_image2.jpg']
labels = [0, 1]  # Example labels
train_dataset = load_dataset(filenames, labels)

# Visualize the dataset pipeline
for image, label in train_dataset.take(1):
    print(image.numpy().shape, label.numpy())
