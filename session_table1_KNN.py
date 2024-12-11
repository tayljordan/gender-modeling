import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
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
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Load a pre-trained CNN model or a feature extractor
cnn_model = load_model(model_path)

# Rebuild the model with an explicit input layer
inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
outputs = cnn_model(inputs)  # Pass inputs through the model to establish the layers
cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Extract features from an appropriate layer
feature_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

# Generate features for KNN
train_features = feature_extractor.predict(train_generator, verbose=1)
val_features = feature_extractor.predict(val_generator, verbose=1)

# Flatten the features
train_features = train_features.reshape(train_features.shape[0], -1)
val_features = val_features.reshape(val_features.shape[0], -1)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Get labels
train_labels = train_generator.classes
val_labels = val_generator.classes

# Train KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn_model.fit(train_features, train_labels)

# Evaluate KNN
val_predictions = knn_model.predict(val_features)
print(f"KNN Validation Accuracy: {accuracy_score(val_labels, val_predictions):.4f}")
print("Validation Classification Report:")
print(classification_report(val_labels, val_predictions))

# Training accuracy (optional, for comparison)
train_predictions = knn_model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f"KNN Training Accuracy: {train_accuracy:.4f}")
