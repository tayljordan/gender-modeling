# Gender Identification in Maritime using TensorFlow and PyTorch

This project implements gender identification models tailored to the maritime industry using TensorFlow and PyTorch. The goal is to analyze maritime-related imagery to address gender disparity in the maritime sector.

---

## Collaborators

<p align="center">
  <img src="static/amet-logo.png" alt="AMET Logo" width="30%">
&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="static/si_logo.png" alt="SI Logo" width="30%">
</p>

## Co-Authors (alphabetical)

[Dr. Padmapriya Jayaraman](https://github.com/padmapriyajayaraman)  
Dr. T. Sasilatha  

---

## Features

1. **Dynamic Hyperparameter Tuning**  
   - Utilizes Keras Tuner for searching optimal hyperparameters.  

2. **Transfer Learning**  
   - Base feature extraction with MobileNetV2.  

3. **Augmented Training**  
   - Incorporates rotation, zoom, and brightness adjustment for robustness.

4. **Regularization Techniques**  
   - Implements dropout and L2 regularization.

5. **False Image Analysis**  
   - Misclassified images are stored for detailed analysis.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets

Organize your dataset in the following structure:
```
data/
    female/
        image1.jpg
        image2.jpg
    male/
        image1.jpg
        image2.jpg
```

---

## Usage

### 1. Update Configuration

Edit the `config.yaml` file to define parameters such as image size, learning rate, and dataset paths.

### 2. Run Training

```bash
python train.py
```

### 3. Evaluate Model

Evaluate the model and analyze results:
```bash
python test.py
```

---

## Key Highlights

- **Hyperparameter Search**  
  Bayesian Optimization via Keras Tuner dynamically determines the best configuration.

- **Misclassification Handling**  
  Copies false positives and false negatives into directories for debugging.

- **Extensive Augmentation**  
  Enhances dataset variability using transformations like rotation and brightness adjustment.

---

## Visualization

Monitor training and validation using TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## Requirements

- Python 3.8 or later  
- TensorFlow 2.6 or later  
- PyYAML  
- NumPy  

---

## Notes

- Misclassified images are stored in `false-images/female_false` and `false-images/male_false`.
- The current implementation uses RGB images. Update input shape to `(height, width, 1)` for grayscale.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

