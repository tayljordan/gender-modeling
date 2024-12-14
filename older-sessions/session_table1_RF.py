import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load configuration
config_path = "/Users/jordantaylor/PycharmProjects/gender-modeling/config_grind.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

params = config['parameters']
image_size = tuple(params['image_size'])
batch_size = params['batch_size']
validation_split = params['validation_split']
rescale = params['rescale']
data_dir = config['directories']['data_dir']
model_path = config['directories']['model_path']

# Data generators for CNN feature extraction
if config['data_generators']['enable_augmentation']:
    datagen = ImageDataGenerator(
        rescale=rescale,
        validation_split=validation_split,
        brightness_range=config['data_generators']['augmentation_params']['brightness_range'],
        fill_mode=config['data_generators']['augmentation_params']['fill_mode'],
        height_shift_range=config['data_generators']['augmentation_params']['height_shift_range'],
        horizontal_flip=config['data_generators']['augmentation_params']['horizontal_flip'],
        rotation_range=config['data_generators']['augmentation_params']['rotation_range'],
        shear_range=config['data_generators']['augmentation_params']['shear_range'],
        width_shift_range=config['data_generators']['augmentation_params']['width_shift_range'],
        zoom_range=config['data_generators']['augmentation_params']['zoom_range']
    )
else:
    datagen = ImageDataGenerator(rescale=rescale, validation_split=validation_split)

train_generator = datagen.flow_from_directory(
    data_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='training', shuffle=False
)
val_generator = datagen.flow_from_directory(
    data_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='validation', shuffle=False
)

# Load pre-trained CNN model
cnn_model = load_model(model_path)

# Rebuild the model with an explicit input layer
# inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
# outputs = cnn_model(inputs)  # Pass inputs through the model to establish the layers
# cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Use an appropriate layer for feature extraction
feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Generate features for Random Forest
train_features = feature_extractor.predict(train_generator, verbose=1)
val_features = feature_extractor.predict(val_generator, verbose=1)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.reshape(train_features.shape[0], -1))
val_features = scaler.transform(val_features.reshape(val_features.shape[0], -1))

# Get labels
train_labels = train_generator.classes
val_labels = val_generator.classes

# Train Random Forest with improved parameters
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
)
rf_model.fit(train_features, train_labels)

# Evaluate Random Forest
val_predictions = rf_model.predict(val_features)
print(f"Random Forest Validation Accuracy: {accuracy_score(val_labels, val_predictions):.4f}")
print("Validation Classification Report:")
print(classification_report(val_labels, val_predictions))

train_predictions = rf_model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f"Random Forest Training Accuracy: {train_accuracy:.4f}")
