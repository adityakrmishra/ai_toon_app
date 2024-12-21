import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    min_delta=0.001,     # Minimum change to qualify as an improvement
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Verbosity mode
    mode='min',          # Mode: 'min', 'max', or 'auto'
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Example model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping
model.fit(train_data, train_labels, epochs=100, validation_data=(val_data, val_labels), callbacks=[early_stopping])
