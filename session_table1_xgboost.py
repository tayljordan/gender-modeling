import os
import yaml
import numpy as np
import xgboost as xgb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler

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

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
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

# Feature extraction using flattened pixel data directly
# Convert images from generators into 2D arrays
train_features = np.concatenate(
    [train_generator[i][0].reshape(len(train_generator[i][1]), -1) for i in range(len(train_generator))], axis=0
)
train_labels = train_generator.classes

val_features = np.concatenate(
    [val_generator[i][0].reshape(len(val_generator[i][1]), -1) for i in range(len(val_generator))], axis=0
)
val_labels = val_generator.classes

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Train XGBoost model using CPU
xgb_model = xgb.XGBClassifier(
    tree_method="hist",  # Use CPU-based histogram method
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

# Train the model
xgb_model.fit(
    train_features, train_labels,
    eval_set=[(val_features, val_labels)],  # Validation data
    verbose=True
)

# Evaluate XGBoost
train_accuracy_xgb = xgb_model.score(train_features, train_labels)
val_accuracy_xgb = xgb_model.score(val_features, val_labels)

print(f"XGBoost Training Accuracy: {train_accuracy_xgb:.4f}")
print(f"XGBoost Validation Accuracy: {val_accuracy_xgb:.4f}")
