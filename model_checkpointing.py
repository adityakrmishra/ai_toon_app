import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the model checkpoint callback
checkpoint = ModelCheckpoint(
    filepath='best_model.h5',  # Path to save the model file
    monitor='val_loss',        # Metric to monitor
    verbose=1,                 # Verbosity mode
    save_best_only=True,       # Save only the best model
    mode='min',                # Mode: 'min', 'max', or 'auto'
    save_weights_only=False,   # Save the full model or only weights
    save_freq='epoch'          # Frequency to save the model
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

# Train the model with checkpointing
model.fit(train_data, train_labels, epochs=100, validation_data=(val_data, val_labels), callbacks=[checkpoint])
