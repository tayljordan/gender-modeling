import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
optimizer_type = params['optimizer']['type'].lower()
learning_rate = params['optimizer']['learning_rate']

# Regularization and model architecture
dense_units = params['dense_units']
dropout_rate = params['dropout_rate']
l2_regularization = params['regularization']['l2']

# Directories
data_dir = "gender-training-dataset"
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

# Data generators (no augmentation)
datagen = ImageDataGenerator(
    rescale=rescale,           # Only rescale the pixel values
    validation_split=validation_split  # Split training and validation data
)

# Training data generator
train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Check class indices
print(f"Class indices: {train_generator.class_indices}")

# Build the model
model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),  # Replace Flatten with GlobalAveragePooling

    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])


# Add dense layers
for units in dense_units:
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(layers.Dropout(dropout_rate))

# Output layer
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
if optimizer_type == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
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
        min_lr=float(lr_scheduler_params['min_lr'])
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
