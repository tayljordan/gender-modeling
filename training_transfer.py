# Imports (remove duplicate callback imports)
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import backend as K

class DynamicDropoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_indices, initial_rate, reduction_factor, reduction_epoch):
        """
        Parameters:
        - layer_indices: List of layer indices where dropout will be adjusted.
        - initial_rate: Initial dropout rate.
        - reduction_factor: Factor by which the dropout rate will be reduced.
        - reduction_epoch: Epoch after which the reduction occurs.
        """
        self.layer_indices = layer_indices
        self.initial_rate = initial_rate
        self.reduction_factor = reduction_factor
        self.reduction_epoch = reduction_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.reduction_epoch:
            new_rate = self.initial_rate * self.reduction_factor
            for i in self.layer_indices:
                try:
                    layer = self.model.layers[i]
                    if isinstance(layer, (layers.Dropout, layers.SpatialDropout2D)):
                        layer._rate = tf.Variable(new_rate, trainable=False, dtype=tf.float32)  # Safely update rate
                        print(f"Adjusted dropout rate for layer {i} to {new_rate:.4f}")
                except Exception as e:
                    print(f"Warning: Could not adjust dropout rate for layer {i}. Error: {e}")


# Load configuration
config_path = "/Users/jordantaylor/PycharmProjects/gender-modeling/config_grind.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters
params = config['parameters']
dense1_units, dense2_units = params['dense1_units'], params['dense2_units']
dropout3, dropout4 = params['dropout3'], params['dropout4']
learning_rate = params['learning_rate']
image_size = tuple(params['image_size'])
batch_size, epochs, validation_split = params['batch_size'], params['epochs'], params['validation_split']
data_dir, model_path = config['directories']['data_dir'], config['directories']['model_path']
l1, l2 = params['l2_regularization']['l1'], params['l2_regularization']['l2']

# Data Generators
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocessing for ResNet50
    validation_split=validation_split,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Update Data Generators to produce 128x128 images
train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),  # Set target size to 128x128
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(128, 128),  # Set target size to 128x128
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load ResNet50 base model with modified input shape
base_model = ResNet50(
    weights='imagenet',  # Use pre-trained ImageNet weights
    include_top=False,  # Exclude the top (fully connected) layers
    input_shape=(128, 128, 3)  # Adjusted input shape to 128x128
)

# Freeze the base model layers
base_model.trainable = False

# Define the model using ResNet50 as a feature extractor
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=dense1_units, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3)),
    layers.Dropout(dropout3),
    layers.Dense(units=dense2_units, activation='relu',
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
    layers.Dropout(dropout4),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define Callbacks
dropout_callback = DynamicDropoutCallback(
    layer_indices=[], initial_rate=0.5, reduction_factor=0.5, reduction_epoch=10
)
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-7,
    cooldown=2,
    verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)

callbacks = [early_stopping, model_checkpoint, lr_scheduler, dropout_callback]

# Train the model (Initial training with frozen ResNet50)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,  # Initial training epochs
    callbacks=callbacks
)

# Evaluate after initial training
initial_loss, initial_accuracy = model.evaluate(val_generator, verbose=1)
print(f"Initial Training Loss: {initial_loss:.4f}, Initial Training Accuracy: {initial_accuracy:.4f}")

# ---- Start Fine-Tuning ----
# Unfreeze the last few layers of ResNet50
for layer in base_model.layers[-10:]:  # Adjust the number of layers to unfreeze
    layer.trainable = True

# Recompile the model with a smaller learning rate
fine_tune_learning_rate = 1e-5
model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 10  # Number of epochs for fine-tuning
history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=fine_tune_epochs,
    callbacks=callbacks
)

# Evaluate the fine-tuned model
fine_tune_loss, fine_tune_accuracy = model.evaluate(val_generator, verbose=1)
print(f"Fine-Tuning Loss: {fine_tune_loss:.4f}, Fine-Tuning Accuracy: {fine_tune_accuracy:.4f}")

# ---- End Fine-Tuning ----

# Save the final model after fine-tuning
model.save(model_path)

# Final evaluation metrics (can include confusion matrix, etc.)
# (Retain your existing metric calculation code here)


# Evaluate Metrics
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Predict on the validation set
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

# Print metrics
print("\nPerformance Metrics:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2%}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1_score:.2%}")

# Summary tables
import pandas as pd

# Table 1: Accuracy comparison
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
accuracy_table = pd.DataFrame({
    "Model": ["Current Model"],
    "Training Accuracy (%)": [train_accuracy * 100],
    "Validation Accuracy (%)": [val_accuracy * 100]
})
print("\nAccuracy Comparison:")
print(accuracy_table)

# Table 2: Detailed performance metrics
metrics_table = pd.DataFrame({
    "Metric": ["True Positives", "False Positives", "True Negatives", "False Negatives",
               "Precision (%)", "Accuracy (%)", "Recall (%)", "F1 Score (%)"],
    "Value": [tp, fp, tn, fn, precision * 100, accuracy * 100, recall * 100, f1_score * 100]
})
print("\nDetailed Performance Metrics:")
print(metrics_table)
