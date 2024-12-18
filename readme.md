# Gender Identification in Maritime using TensorFlow

This project aims to reduce bias, promote fairness, and highlight underrepresentation in the maritime industry by developing gender identification models using TensorFlow. By analyzing maritime-related imagery, the project addresses the gender disparity and brings attention to the limited visibility of women in this traditionally male-dominated sector. Through the innovative use of computer vision technology, the goal is to foster a more equitable and inclusive representation of female professionals in maritime.

---

## Partner Institutions

<p align="center">
  <img src="static/amet-logo.png" alt="AMET Logo" width="30%">
&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="static/si_logo.png" alt="SI Logo" width="30%">
</p>

## Collaborators (alphabetical)

[Dr. Padmapriya Jayaraman](https://github.com/padmapriyajayaraman)  
Dr. T. Sasilatha  

---

## Features

1. **Dynamic Hyperparameter Tuning**  
   - Uses Keras Tuner to identify optimal hyperparameter configurations for model training. This allows the model to achieve better accuracy by exploring a wide range of possible configurations efficiently.

2. **Transfer Learning**  
   - Leverages MobileNetV2 for efficient base feature extraction, reducing the time and computational resources needed for training while maintaining high performance.

3. **Augmented Training**  
   - Applies data augmentation techniques like rotation, zoom, brightness adjustments, and flips to enhance model generalization and ensure robustness against real-world variations.

4. **Regularization Techniques**  
   - Implements advanced techniques such as dropout and L2 regularization to prevent overfitting and maintain model generalization across diverse datasets.

5. **False Image Analysis**  
   - Stores misclassified images in designated directories for further analysis, enabling developers to identify patterns in errors and refine the model for improved accuracy.

6. **Data Diversity**  
   - Ensures datasets are representative of the diverse maritime workforce, incorporating images from various ethnic backgrounds and industrial contexts to minimize biases.

---

## Setup Instructions

### 1. Clone the Repository

Begin by cloning the repository to your local system:
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

Create and activate a virtual environment to manage project dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

Install the required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets

Organize your image dataset in the following structure for training and validation:
```
data/
    female/
        image1.jpg
        image2.jpg
    male/
        image1.jpg
        image2.jpg
```

The dataset should include a balanced mix of images across genders, with annotations reflecting the diversity of the maritime industry.

---

## Usage

### 1. Update Configuration

Customize the `config.yaml` file to set key parameters such as image size, learning rate, batch size, and dataset paths. These configurations allow for flexibility and adaptation to different datasets and training environments.

### 2. Run Training

Train the model using the command:
```bash
python train.py
```

The training process will utilize the specified parameters in `config.yaml` and include real-time feedback on accuracy and loss metrics.

### 3. Evaluate Model

Evaluate the trained model and analyze the results:
```bash
python test.py
```

This step provides insights into the model's performance and highlights areas for further refinement.

---

## Key Highlights

- **Hyperparameter Optimization**  
  - Dynamically determines the best configurations through Bayesian optimization using Keras Tuner, reducing the manual effort in parameter selection and improving model outcomes.

- **Misclassification Debugging**  
  - Provides a mechanism to identify and analyze false positives and false negatives, offering actionable insights for improving model performance.

- **Extensive Data Augmentation**  
  - Introduces robust variability to the dataset through transformations such as rotations, zooms, flips, and brightness adjustments, ensuring better model resilience.

- **Cross-Domain Applicability**  
  - The methodologies employed in this project can be adapted to other industries facing similar challenges with underrepresentation and bias.

---

## Visualization

Track and monitor the training and validation process using TensorBoard. This tool provides real-time visualizations of key metrics:
```bash
tensorboard --logdir logs/
```

TensorBoard graphs help identify trends in accuracy, loss, and other performance metrics, enabling data-driven model refinement.

---

## Requirements

- Python 3.8 or later  
- TensorFlow 2.6 or later  
- PyYAML for configuration management  
- NumPy for numerical computations  
- TensorBoard for visualization  

---

## Notes

- Misclassified images are saved in `false-images/female_false` and `false-images/male_false` for detailed analysis and debugging.
- The implementation currently supports RGB images; modify the input shape to `(height, width, 1)` for grayscale support if required.
- Future updates may include support for additional languages and improved preprocessing pipelines to handle edge cases more effectively.

---

## License

This project is released under the MIT License. Refer to the `LICENSE` file for more details. By adopting this open-source approach, the project encourages collaboration and innovation within the maritime and technology communities.

