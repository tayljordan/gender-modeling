import os
import yaml
import csv
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import warnings

# GPU memory growth setting
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPUs available: {[device.name for device in physical_devices]}")
else:
    print("No GPUs available. Training will run on CPU.")

# Suppress warnings and logs
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer.*")
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
batch_size = max(8, params['batch_size'] // 2)  # Reduce batch size to save memory
epochs = params['epochs']
rescale = params['rescale']
validation_split = params['validation_split']

# Directories
data_dir = config['directories']['data_dir']
model_path = config['model']['path']

# Data generators
datagen = ImageDataGenerator(
    rescale=rescale,
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

# Define the model builder function for Keras Tuner
def build_model(hp):
    model = models.Sequential([
        layers.Input(shape=(image_size[0], image_size[1], 3)),
        layers.Conv2D(
            filters=hp.Int("conv1_filters", min_value=32, max_value=64, step=32),
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(hp.Float("dropout1", 0.1, 0.3, step=0.1)),

        layers.Conv2D(
            filters=hp.Int("conv2_filters", min_value=64, max_value=128, step=64),
            kernel_size=(3, 3),
            kernel_initializer='he_normal',
            padding='same'
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(hp.Float("dropout2", 0.1, 0.3, step=0.1)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            units=hp.Int("dense1_units", min_value=64, max_value=256, step=64),
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(hp.Float("dropout3", 0.1, 0.3, step=0.1)),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
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
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
]

# Perform hyperparameter search
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=8,  # Reduce search epochs to minimize memory usage
    callbacks=callbacks
)

# Get the best hyperparameters and best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

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
