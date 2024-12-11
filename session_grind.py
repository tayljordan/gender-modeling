import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load configuration from YAML file
config_path = "/Users/jordantaylor/PycharmProjects/gender-modeling/config_grind.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from YAML
params = config['parameters']
conv1_filters = params['conv1_filters']
dropout1 = params['dropout1']
conv2_filters = params['conv2_filters']
dropout2 = params['dropout2']
conv3_filters = params['conv3_filters']
dense1_units = params['dense1_units']
dropout3 = params['dropout3']
dense2_units = params['dense2_units']
dropout4 = params['dropout4']
learning_rate = params['learning_rate']

# Extract training configurations
image_size = tuple(params['image_size'])
batch_size = params['batch_size']
epochs = params['epochs']
validation_split = params['validation_split']

data_dir = config['directories']['data_dir']
model_path = config['directories']['model_path']

# Extract L2 regularization parameters
l1 = params['l2_regularization']['l1']
l2 = params['l2_regularization']['l2']

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Build the model with L2 regularization
model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(filters=conv1_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(dropout1),

    layers.Conv2D(filters=conv2_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(dropout2),

    layers.Conv2D(filters=conv3_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.GlobalAveragePooling2D(),

    # Dense layers with L2 regularization
    layers.Dense(units=dense1_units, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Dropout(dropout3),

    layers.Dense(units=dense2_units, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Dropout(dropout4),

    layers.Dense(1, activation='sigmoid')
])

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=8,  # Increased from 5 to 8
    min_lr=1e-6,
    verbose=1
)


# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True),
    lr_scheduler  # Add the learning rate scheduler
]


# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Save the trained model
model.save(model_path)
