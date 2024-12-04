import os
import yaml
import csv
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

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

# Optimizer parameters
optimizer_type = params['optimizer']['type'].lower()
learning_rate = params['optimizer']['learning_rate']

# Regularization
dense_units = params['dense_units']
dropout_rate = params['dropout_rate']
l1_regularization = params['regularization']['l1']
l2_regularization = params['regularization']['l2']

# Directories
data_dir = config['directories']['data_dir']

model_path = config['model']['path']

# Callbacks configuration
checkpoint_params = config['model']['checkpoint']
lr_scheduler_params = params['lr_scheduler']

# GPU check
gpus = tf.config.list_physical_devices('GPU')
gpu_status = f"GPUs are available: {len(gpus)} GPU(s) detected." if gpus else "No GPUs detected. Using CPU."
print(gpu_status)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Data generators with augmentation toggling
datagen = ImageDataGenerator(
    rescale=rescale,
    validation_split=validation_split
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

# Log total images
total_train_images = train_generator.samples
total_val_images = val_generator.samples
with open(os.path.join(log_dir, "metadata.txt"), "w") as metadata_file:
    metadata_file.write(f"{gpu_status}\n")
    metadata_file.write(f"Total training images: {total_train_images}\n")
    metadata_file.write(f"Total validation images: {total_val_images}\n")


lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,  # Decay every 1000 steps
    decay_rate=0.96,  # Reduce LR by 4% each step
    staircase=True
)


# Build the model
model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(0.2),

    layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(0.2),

    layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.GlobalAveragePooling2D(),

    layers.Dense(dense_units[0], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)),
    layers.Dropout(dropout_rate),
    layers.Dense(dense_units[1], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)),
    layers.Dropout(dropout_rate),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
if optimizer_type == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
elif optimizer_type == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
else:
    raise ValueError("Unsupported optimizer type.")

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_params['filepath'], monitor='val_loss', save_best_only=True),

]

if lr_scheduler_params.get('enable', False):
    callbacks.append(ReduceLROnPlateau(
        monitor=lr_scheduler_params['monitor'],
        factor=lr_scheduler_params['factor'],
        patience=lr_scheduler_params['patience'],
        min_lr=float(lr_scheduler_params['min_lr'])
    ))

# CSV logging setup
csv_path = os.path.join(log_dir, "training_log.csv")
with open(csv_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])

# Training with epoch logging
class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(csv_path, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, logs.get("accuracy"), logs.get("loss"),
                                 logs.get("val_accuracy"), logs.get("val_loss")])

callbacks.append(CSVLoggerCallback())

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
