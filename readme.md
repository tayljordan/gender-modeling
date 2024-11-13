# Gender Identification in Maritime using TensorFlow

This project demonstrates the implementation of a neural network using TensorFlow for gender identification in the maritime industry. The focus is on developing a model to analyze maritime-related imagery, contributing to research and solutions for gender disparity in the sector.

## Requirements

- Python 3.11+
- TensorFlow
- certifi

## Setup

1. Install the required packages (Mac only):
   ```bash
   pip install tensorflow certifi
   ```
2. Ensure the environment is configured for SSL by setting the appropriate certificate file:
   ```python
   import os
   import ssl
   import certifi

   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
   os.environ['SSL_CERT_FILE'] = certifi.where()
   ssl._create_default_https_context = ssl.create_default_context
   ```

## Dataset

The dataset used for this project consists of publicly sourced maritime-related images. These images are analyzed to identify gender markers, employment roles (e.g., at sea or ashore), and geographic origins. This development phase serves as a foundational step for building robust models tailored to the maritime sector.

## Implementation

### 1. Load and Normalize the Data
```python
# Placeholder for loading maritime-related image data
(training_images, training_labels), (test_images, test_labels) = load_maritime_dataset()

# Normalize the images to a range of 0-1
training_images = training_images / 255.0
test_images = test_images / 255.0
```

### 2. Design the Model

A simple feedforward neural network with:
- A flattening layer
- A dense hidden layer with ReLU activation
- An output layer with softmax activation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### 3. Compile and Train the Model

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```

### 4. Evaluate the Model

```python
model.evaluate(test_images, test_labels)
```

## Future Work

This project aims to refine the model by utilizing a dataset comprising 1,200 maritime-related images collected from governmental and non-governmental organizations worldwide. The focus will include gender identification, role classification, and geographic analysis to address systemic challenges in gender inclusivity within the maritime industry.

## Supporting Research

This work aligns with ongoing efforts to address gender disparity in the maritime industry. By integrating data-driven methodologies, such as vision transformers and natural language models, this project aims to highlight and promote gender equality in alignment with global initiatives like the United Nations 2030 Agenda for Sustainable Development (Goal 5). A more inclusive maritime sector can be achieved by leveraging data analysis and AI tools to visualize current trends and challenges.

### Key References
1. Kitada, M., et al. (2022)
2. Dragomir & Senbursa (2019)
3. Buolamwini, J., et al. (2018)

This study incorporates image preprocessing, visual embedding, and OpenAI’s GPT-4 API to explore and mitigate gender bias in maritime-related imagery. Learn more about the methodology and its application in the context of gender equality in the maritime industry.

