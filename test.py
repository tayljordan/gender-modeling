import os
import yaml
import numpy as np
import xgboost as xgb
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout

# Load configuration from YAML file
config_path = "/Users/jordantaylor/PycharmProjects/gender-modeling/config_grind.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters
params = config['parameters']
image_size = tuple(params['image_size'])
batch_size = params['batch_size']
validation_split = params['validation_split']

data_dir = config['directories']['data_dir']

# Data generators with augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split,
    **config['data_generators']['augmentation_params']
)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True  # Enable shuffling for training data
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Load and rebuild the model with dropout
loaded_model = models.load_model(config['directories']['model_path'])
input_layer = Input(shape=(image_size[0], image_size[1], 3))
x = loaded_model(input_layer)
x = Dropout(rate=params.get('dropout4', 0.5))(x)  # Add dropout for regularization
feature_extractor = Model(inputs=input_layer, outputs=x)

# Ensure penultimate layer features are used correctly
feature_extractor = Model(
    inputs=feature_extractor.input,
    outputs=feature_extractor.layers[-2].output  # Use the penultimate layer
)

# Extract features and flatten
train_features = feature_extractor.predict(train_generator, verbose=1)
train_features = train_features.reshape(train_features.shape[0], -1)
train_labels = train_generator.classes

val_features = feature_extractor.predict(val_generator, verbose=1)
val_features = val_features.reshape(val_features.shape[0], -1)
val_labels = val_generator.classes

# Train XGBoost model with early stopping
xgb_model = xgb.XGBClassifier(
    tree_method="hist",  # Use CPU-based histogram method
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=500,  # Increase trees for better learning
    learning_rate=0.01,  # Reduce learning rate
    max_depth=6,  # Improve depth for feature interaction
    reg_alpha=0.0008,  # L1 regularization from YAML
    reg_lambda=0.001  # L2 regularization from YAML
)

xgb_model.fit(
    train_features, train_labels,
    eval_set=[(val_features, val_labels)],
    early_stopping_rounds=20,  # Stop training early if no improvement
    verbose=True
)

# Evaluate XGBoost
train_predictions = xgb_model.predict(train_features)
val_predictions = xgb_model.predict(val_features)

train_accuracy_xgb = accuracy_score(train_labels, train_predictions)
val_accuracy_xgb = accuracy_score(val_labels, val_predictions)

print(f"XGBoost Training Accuracy: {train_accuracy_xgb:.4f}")
print(f"XGBoost Validation Accuracy: {val_accuracy_xgb:.4f}")
