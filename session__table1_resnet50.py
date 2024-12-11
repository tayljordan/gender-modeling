import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt

# Load configuration from YAML file
config_path = "/Users/jordantaylor/PycharmProjects/gender-modeling/config_grind.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from YAML
params = config['parameters']
batch_size = params['batch_size']
epochs = params['epochs']
validation_split = params['validation_split']
learning_rate = params['learning_rate']

# Directory paths
data_dir = config['directories']['data_dir']
model_path = config['directories']['model_path']

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),  # ResNet50 expects 224x224 images
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ResNet50 Model with Transfer Learning
base_model = applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base model layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),  # Custom dense layers
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True),
    lr_scheduler
]

# Train the ResNet50 model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the ResNet50 model
train_acc = history.history['accuracy'][-1] * 100
val_acc = history.history['val_accuracy'][-1] * 100
accuracy_gap = train_acc - val_acc
print(f"Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%, Difference: {accuracy_gap:.2f}%")

# Save model results for comparison
comparison_data = [
    {"Model": "ResNet50", "Training Accuracy (%)": train_acc, "Validation Accuracy (%)": val_acc, "Difference (%)": accuracy_gap}
]

# Convert results into a DataFrame
comparison_table = pd.DataFrame(comparison_data)

# Display comparison table
print("\nTable 1: Model Comparison")
print(comparison_table)

# Visualization
comparison_table.plot(
    x='Model',
    y=['Training Accuracy (%)', 'Validation Accuracy (%)'],
    kind='bar',
    figsize=(8, 5)
)
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison: Training vs Validation Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
