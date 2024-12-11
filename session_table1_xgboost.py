import os
import yaml
import numpy as np
import xgboost as xgb
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


from tensorflow.keras import Input, Model

# Load the trained Sequential model
loaded_model = models.load_model("/Users/jordantaylor/PycharmProjects/gender-modeling/models/best_gender_model_v2.keras")

# Rebuild as a functional model
input_layer = Input(shape=(image_size[0], image_size[1], 3))
outputs = loaded_model(input_layer)  # Call the Sequential model
feature_extractor = Model(inputs=input_layer, outputs=outputs)

# Extract features from the penultimate layer
feature_extractor = Model(
    inputs=feature_extractor.input,  # Input from the rebuilt model
    outputs=feature_extractor.layers[-2].output  # Use the penultimate layer for features
)



# Extract features
train_features = feature_extractor.predict(train_generator, verbose=1)
train_labels = train_generator.classes

val_features = feature_extractor.predict(val_generator, verbose=1)
val_labels = val_generator.classes

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    tree_method="gpu_hist",  # Use GPU if available
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(
    train_features, train_labels,
    eval_set=[(val_features, val_labels)],
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate XGBoost
train_accuracy_xgb = xgb_model.score(train_features, train_labels)
val_accuracy_xgb = xgb_model.score(val_features, val_labels)

print(f"XGBoost Training Accuracy: {train_accuracy_xgb:.4f}")
print(f"XGBoost Validation Accuracy: {val_accuracy_xgb:.4f}")

# Evaluate CNN directly
cnn_loss, cnn_accuracy = feature_extractor.evaluate(val_generator, verbose=1)
print(f"Custom CNN Validation Accuracy: {cnn_accuracy:.4f}")
