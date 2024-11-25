import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config
params = config['parameters']
image_size = tuple(params['image_size'])  # Input size of the images
batch_size = params['batch_size']  # Batch size
epochs = params['epochs']  # Number of epochs for each training phase
rescale = params['rescale']  # Rescaling factor
learning_rate = params['optimizer']['learning_rate']  # Learning rate for initial training
fine_tune_lr = learning_rate / 10  # Lower learning rate for fine-tuning
dense_units = params['dense_units']  # Dense layer units
dropout_rate = params['dropout_rate']  # Dropout rate
l2_regularization = params['regularization']['l2']  # L2 regularization factor

# Directories
female_dir = os.path.join(os.getcwd(), config['directories']['female_dir'])
male_dir = os.path.join(os.getcwd(), config['directories']['male_dir'])
tensorboard_log_dir = os.path.join(os.getcwd(), config['logs']['tensorboard']['log_dir'])
model_path = os.path.join(os.getcwd(), config['model']['path'])

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
    rescale=rescale,
    validation_split=params['validation_split']  # Split training and validation data
)

train_generator = datagen.flow_from_directory(
    directory=os.path.dirname(female_dir),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=os.path.dirname(female_dir),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# MobileNetV2 with pretrained weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(image_size[0], image_size[1], 3),
    include_top=False,
    weights='imagenet'  # Use pretrained weights
)

# Freeze the base model initially
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(dropout_rate),
    layers.Dense(dense_units[0], activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)),
    layers.Dropout(dropout_rate),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model for the initial training phase
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor=lr_scheduler_params['monitor'],
    factor=lr_scheduler_params['factor'],
    patience=lr_scheduler_params['patience'],
    min_lr=lr_scheduler_params['min_lr']
)
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_params['filepath'],
    monitor=checkpoint_params['monitor'],
    save_best_only=checkpoint_params['save_best_only']
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_log_dir,
    histogram_freq=config['logs']['tensorboard']['histogram_freq']
)

callbacks = [early_stopping, reduce_lr, model_checkpoint, tensorboard_callback]

# Train only the top layers
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Fine-tune the base model
base_model.trainable = True

# Recompile the model for the fine-tuning phase
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training with the base model unfrozen
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Save the final model
model.save(model_path)
