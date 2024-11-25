# Gender Identification in Maritime using TensorFlow and PyTorch

This project implements gender identification models tailored to the maritime industry using TensorFlow and PyTorch. The goal is to analyze maritime-related imagery to address gender disparity in the maritime sector.

---

## Collaborators:

<p align="center">
  <img src="static/amet-logo.png" alt="AMET Logo" width="30%">
&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="static/si_logo.png" alt="SI Logo" width="30%">
</p>

## Co-Authors (alphabetical):

[Dr. Padmapriya Jayaraman](https://github.com/padmapriyajayaraman)

Dr. T. Sasilatha

---

## **Features**
1. **Transfer Learning**:
   - Uses a pre-trained MobileNetV2 base for feature extraction.
   - Fine-tunes the base model after training the custom layers.
   
2. **Dynamic Configuration**:
   - All key parameters (e.g., image size, learning rate, batch size) are adjustable via a `config.yaml` file.
   - Facilitates rapid experimentation without modifying the codebase.

3. **Callbacks**:
   - Early stopping to avoid overfitting.
   - Learning rate adjustment using `ReduceLROnPlateau`.
   - TensorBoard logging for detailed visualization.

4. **Regularization**:
   - L2 regularization and dropout are applied to mitigate overfitting.

5. **Reproducibility**:
   - Configurable random seed for consistent results.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-name>
```

### **2. Install Dependencies**
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **3. Prepare Datasets**
Place your datasets in the following directory structure:
```
gender-training-dataset/
    female_augmented/
        image1.jpg
        image2.jpg
    male_augmented/
        image1.jpg
        image2.jpg
test-set-female/
    test_image1.jpg
    test_image2.jpg
test-set-male/
    test_image1.jpg
    test_image2.jpg
```

---

## **Usage**

### **1. Update Configuration**
Edit the `config.yaml` file to customize the parameters:
```yaml
parameters:
  image_size: [64, 64]
  batch_size: 32
  epochs: 10
  rescale: 1.0
  validation_split: 0.2
  optimizer:
    type: "Adam"
    learning_rate: 0.001
  lr_scheduler:
    enable: true
    monitor: "val_loss"
    factor: 0.2
    patience: 3
    min_lr: 1e-6
directories:
  female_dir: "gender-training-dataset/female_augmented"
  male_dir: "gender-training-dataset/male_augmented"
  test_set_male: "test-set-male"
  test_set_female: "test-set-female"
logs:
  tensorboard_log_dir: "logs/gender_model_v1"
model:
  path: "best_gender_model.keras"
  checkpoint:
    enable: true
    filepath: "best_gender_model.keras"
    monitor: "val_loss"
    save_best_only: true
```

### **2. Train the Model**
Run the training script:
```bash
python train.py
```

### **3. Evaluate the Model**
Evaluate the model using your test sets:
```bash
python test.py
```

---

## **Key Components**

### **Configuration (`config.yaml`)**
All key training and testing parameters are defined in `config.yaml`:
- Model architecture and optimization settings.
- Dataset paths and preprocessing steps.
- Callback settings for checkpoints and learning rate reduction.

### **Model Architecture**
The model is built using:
- Pre-trained MobileNetV2 as a base model.
- Additional dense layers with L2 regularization and dropout for robust training.

### **Callbacks**
- **EarlyStopping**: Stops training when validation loss stops improving.
- **ModelCheckpoint**: Saves the best model based on validation performance.
- **ReduceLROnPlateau**: Reduces the learning rate when training stagnates.
- **TensorBoard**: Logs training metrics for visualization.

---

## **Visualization**
Use TensorBoard to monitor training metrics:
```bash
tensorboard --logdir=logs/gender_model_v1
```

---

## **Requirements**
- Python 3.8 or later
- TensorFlow 2.6 or later
- PyYAML for configuration management
- NumPy for numerical computations

---

## **Notes**
- The current implementation uses color images. For grayscale, update the input shape to `(height, width, 1)`.
- Datasets must be preprocessed and augmented before training if additional augmentation is disabled.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.
```

---

### **Highlights of the README**
1. **Comprehensive**:
   - Explains the purpose and features of the project.
   - Provides clear instructions for setup and usage.

2. **Dynamic**:
   - Reflects the YAML-driven configuration for easy adjustments.

3. **Scalable**:
   - Designed to allow future expansion (e.g., grayscale support, other optimizers).

Let me know if you'd like further refinements!