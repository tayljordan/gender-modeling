import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")

import os
import yaml
import datetime
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner as kt

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Timestamp for logs
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/run_{timestamp}"
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
dense_units = params['dense_units']
dropout_rate = params['dropout_rate']
l2_regularization = params['regularization']['l2']

data_dir = config['directories']['data_dir']
model_path = config['model']['path']

# GPU check
gpus = tf.config.list_physical_devices('GPU')
gpu_status = f"GPUs are available: {len(gpus)} GPU(s) detected." if gpus else "No GPUs detected. Using CPU."
print(gpu_status)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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


# Keras Tuner Model Builder
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
    base_model.trainable = False  # Freeze base model

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    for i in range(hp.Int("num_dense_layers", 1, 3)):  # 1-3 Dense Layers
        x = layers.Dense(
            units=hp.Int(f"dense_units_{i}", 64, 512, step=64),  # Tune units: 64-512
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_regularization)
        )(x)
        x = layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", 0.2, 0.5, step=0.1))(x)  # Dropout: 0.2-0.5

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])  # Learning rate: 0.01, 0.001, 0.0001
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Initialize Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory=log_dir,
    project_name="vgg16_nas"
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
]

# Perform Hyperparameter Search
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Get Best Model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate and Save the Best Model
loss, accuracy = best_model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
best_model.save(model_path)
