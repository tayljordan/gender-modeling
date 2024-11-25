import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Load configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config
params = config['parameters']
image_size = tuple(params['image_size'])  # Input image size
batch_size = params['batch_size']  # Batch size
epochs = params['epochs']  # Number of epochs
rescale = params['rescale']  # Rescaling factor
validation_split = params['validation_split']  # Validation split fraction

# Optimizer parameters
optimizer_type = params['optimizer']['type'].lower()  # Ensure lowercase for comparison
learning_rate = params['optimizer']['learning_rate']
momentum = params['optimizer'].get('momentum', 0.0)  # Default to 0.0 if not using SGD

# Regularization and model architecture
dense_units = params['dense_units']  # Dense layer units
dropout_rate = params['dropout_rate']  # Dropout rate
l2_regularization = params['regularization']['l2']  # L2 regularization
l1_regularization = params['regularization']['l1']  # L1 regularization

# Directories
female_dir = config['directories']['female_dir']
male_dir = config['directories']['male_dir']
tensorboard_log_dir = config['logs']['tensorboard']['log_dir']
model_path = config['model']['path']

# Callbacks configuration
checkpoint_params = config['model']['checkpoint']
lr_scheduler_params = params['lr_scheduler']

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs are available: {len(gpus)} GPU(s) detected.")
else:
    print("No GPUs detected. Using CPU.")

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split  # Split training and validation data
)

# Training data generator
train_generator = datagen.flow_from_directory(
    directory=os.path.dirname(female_dir),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    directory=os.path.dirname(female_dir),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Build the model
model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten()
])

# Add dense layers
for units in dense_units:
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(layers.Dropout(dropout_rate))

# Output layer
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
if optimizer_type == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
elif optimizer_type == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_params['filepath'],
    monitor=checkpoint_params['monitor'],
    save_best_only=checkpoint_params['save_best_only']
)
tensorboard_callback = TensorBoard(
    log_dir=tensorboard_log_dir,
    histogram_freq=config['logs']['tensorboard']['histogram_freq']
)

if lr_scheduler_params['enable']:
    reduce_lr = ReduceLROnPlateau(
        monitor=lr_scheduler_params['monitor'],
        factor=lr_scheduler_params['factor'],
        patience=lr_scheduler_params['patience'],
        min_lr=float(lr_scheduler_params['min_lr'])  # Ensure min_lr is a float
    )
    callbacks = [early_stopping, model_checkpoint, tensorboard_callback, reduce_lr]
else:
    callbacks = [early_stopping, model_checkpoint, tensorboard_callback]

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate and print results
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Save the model
model.save(model_path)
