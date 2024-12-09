import os
import yaml
import datetime
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import shutil

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {[device.name for device in physical_devices]}")
else:
    print("No GPUs available. Training will run on CPU.")

# Suppress specific TensorFlow warnings
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer.*"
)

# Suppress specific TensorFlow warnings
warnings.filterwarnings("ignore", message="Your `PyDataset` class should call")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Timestamp for logs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"/Users/jordantaylor/PycharmProjects/gender-modeling/logs/run_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Load configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Save the YAML snapshot in the log directory
yaml_snapshot_path = os.path.join(log_dir, "config_snapshot.yaml")
with open(yaml_snapshot_path, "w") as yaml_file:
    yaml.dump(config, yaml_file)

# Extract parameters from config
params = config['parameters']
image_size = tuple(params['image_size'])
batch_size = params['batch_size']
epochs = params['epochs']
rescale = params['rescale']
validation_split = params['validation_split']

# Directories
data_dir = config['directories']['data_dir']
model_path = config['model']['path']


# No augmentation
'''
datagen = ImageDataGenerator(
    rescale=rescale,
    validation_split=validation_split
)
'''


# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=rescale,
    validation_split=validation_split,
    rotation_range=10,  # Rotate images by 10 degrees
    width_shift_range=0.05,  # Shift images horizontally by 5%
    height_shift_range=0.05,  # Shift images vertically by 5%
    shear_range=0.1,  # Apply shearing transformations
    zoom_range=0.1,  # Apply zooming transformations
    horizontal_flip=True,  # Randomly flip images horizontally
    brightness_range=[0.9, 1.1],  # Adjust brightness
    fill_mode='nearest'  # Fill pixels in new locations
)

# Training generator with augmentation
train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validation generator without augmentation (only rescaling)
val_datagen = ImageDataGenerator(
    rescale=rescale,
    validation_split=validation_split
)

val_generator = val_datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


def build_model(hp):
    model = models.Sequential([
        layers.Input(shape=(image_size[0], image_size[1], 3)),

        # First Conv2D Block
        layers.Conv2D(
            filters=hp.Int("conv1_filters", min_value=16, max_value=64, step=16),
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(hp.Float("dropout1", 0.1, 0.3, step=0.1)),

        # Second Conv2D Block
        layers.Conv2D(
            filters=hp.Int("conv2_filters", min_value=32, max_value=128, step=32),
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        # Dense Layer 1
        layers.Dense(
            units=hp.Int("dense1_units", min_value=32, max_value=128, step=32),
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(hp.Float("dropout3", 0.1, 0.3, step=0.1)),

        # Output Layer
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Initialize Keras Tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=config['keras']['keras_max_trials'],
    directory=log_dir,
    project_name="cnn_tuning"
)

# Callbacks for early stopping and logging
callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, min_delta=0.001, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
]


# Perform hyperparameter search
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=max_epochs,
    callbacks=callbacks
)

# Get the best hyperparameters and best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Update YAML with the best learning rate from the tuner
try:
    params['learning_rate'] = best_hps.get("learning_rate") or params['learning_rate']
except:
    pass

with open(config_path, "w") as file:
    yaml.dump(config, file)

# Recompile the model using the best learning rate from YAML
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

for key, value in best_hps.values.items():
    if key in config['parameters']:
        config['parameters'][key] = value
with open(config_path, "w") as file:
    yaml.dump(config, file)

# Train the best model on the full dataset
history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the model
loss, accuracy = best_model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Save the best model
best_model.save(model_path)


# Later, when loading the model for evaluation or further use:
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile the model with the desired optimizer and learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Evaluate or continue training the model as needed
loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")


# Paths for storing false images
female_false_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/female_false"
male_false_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/male_false"

# Ensure directories exist
os.makedirs(female_false_dir, exist_ok=True)
os.makedirs(male_false_dir, exist_ok=True)

# Generate predictions for validation set
predictions = best_model.predict(val_generator, verbose=1)
ground_truth = val_generator.classes
class_indices = val_generator.class_indices  # Class index mapping

# Convert predictions to binary based on a threshold (0.5 by default)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int).flatten()

# Reverse class mapping for clarity
class_names = {v: k for k, v in class_indices.items()}

# Identify and copy false positives and false negatives
for i, (image_path, pred, true_label) in enumerate(zip(val_generator.filepaths, binary_predictions, ground_truth)):
    # False Negative: Predicted 0 (female) but true is 1 (male)
    if pred == 0 and true_label == 1:
        shutil.copy(image_path, male_false_dir)

    # False Positive: Predicted 1 (male) but true is 0 (female)
    elif pred == 1 and true_label == 0:
        shutil.copy(image_path, female_false_dir)

print("False negatives and false positives have been copied to their respective directories.")
