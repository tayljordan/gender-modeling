# Gender Identification in Maritime using TensorFlow and PyTorch

This project implements gender identification models tailored to the maritime industry using TensorFlow and PyTorch. The goal is to analyze maritime-related imagery to address gender disparity in the maritime sector.

---

## Collaborators:

<p align="center">
  <img src="static/amet-logo.png" alt="AMET Logo" width="30%">
&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="static/si_logo.png" alt="SI Logo" width="30%">
</p>

## Co-Author:

[Dr. Padmapriya Jayaraman](https://github.com/padmapriyajayaraman)

---

## Requirements:

- Python 3.11+
- TensorFlow
- PyTorch
- OpenCV
- NumPy

---

## Dataset:

- **Directory Structure**:
  ```
  gender-dataset/
  ├── female/
  └── male/
  ```
- Each subdirectory contains labeled images for training and testing.

---

## Supporting Research:

This project aligns with global initiatives like the United Nations Sustainable Development Goal 5 to promote gender equality. Leveraging AI tools, this project aims to visualize gender trends and drive inclusivity in the maritime sector.


### Top 3 Factors 

1. **Dataset Composition and Representation**:
   - **Why**: The quality and balance of the dataset are critical for reducing bias and ensuring the model generalizes well.
   - **Actions**:
     - Ensure **equal gender representation** (50% male, 50% female).
     - Include **images with safety gear** (hardhat, safety glasses, etc.) to mitigate visual bias like associating hardhats with males.
     - Address **ethnic diversity** to avoid biases toward dominant groups (e.g., Filipino, Indian).
     - Maintain **uniform resolution** and include augmentations like flipping, rotations, and brightness adjustments.

2. **Image Preprocessing Changes**:
   - **Why**: The current setup (28x28 grayscale images) may oversimplify features, potentially missing critical details like gear or subtle facial features.
   - **Actions**:
     - Increase image size (e.g., **64x64 or 128x128**) to capture more visual context.
     - Use **color images** (cv2.IMREAD_COLOR) to include richer visual information (e.g., reflective tape, uniform colors).
     - Normalize appropriately based on the new input (e.g., [0, 1] for color).

3. **Data Augmentation**:
   - **Why**: Augmentation increases dataset variability, helping the model generalize to unseen images.
   - **Actions**:
     - Implement augmentations like random cropping, rotation, flipping, and brightness adjustments to account for background complexity and variability in safety gear.

### **To-Do List for Improving and Benchmarking the Model**

#### **Dataset (Most Important)**
1. **Ensure Balanced Gender Representation**:
   - Create a dataset with 50% male and 50% female images to ensure fairness and reduce gender bias.

2. **Expand Sample Size**:
   - Increase the number of training images to improve generalization and accuracy.

3. **Address Ethnic Diversity**:
   - Include images representative of major seafarer ethnicities: Filipino, Indian, Chinese/Japanese, and Eastern European.

4. **Incorporate Peripheral Safety Gear**:
   - Include images where subjects wear hardhats, safety glasses, reflective tape, or coveralls to simulate real-world conditions.

5. **Tackle Background Complexity**:
   - Use images with a mix of simple and complex backgrounds to make the model more robust.

6. **Standardize Box Size (Face Frame)**:
   - Adjust bounding boxes (e.g., using InsightFace) to include some peripheral context (safety gear, neck, and shoulders). Train and compare the performance.

7. **Uniform Image Resolution**:
   - Ensure all images have the same resolution before training to maintain consistency.

8. **Augment Data**:
   - Use augmentation techniques (`augment.py`) such as flipping, rotation, brightness adjustment, and random cropping to artificially expand the dataset.

9. **Address Age Distribution**:
   - Ensure the dataset has a balance of age groups to prevent bias toward younger women (<40 years old) or men.

---

#### **Preprocessing**
10. **Increase Image Resizing Dimensions**:
    - Change resizing from `28x28` to higher dimensions (e.g., `64x64` or `128x128`) to capture more features.

11. **Use Color Images**:
    - Replace grayscale (e.g., `cv2.IMREAD_GRAYSCALE`) with color images (`cv2.IMREAD_COLOR`) to incorporate richer visual information.

12. **Adjust Normalization Ranges**:
    - Experiment with `[0,1]` and `[-1,1]` normalization ranges to determine the best preprocessing scale.

13. **Adjust Normalization Parameters**:
    - Fine-tune `mean` and `std` values in `transforms.Normalize` to better align with the dataset's pixel distribution.

14. **Toggle DataLoader Shuffle**:
    - Experiment with `shuffle=True` or `shuffle=False` to observe the impact on training consistency.

---

#### **Training**
15. **Add More Convolutional Layers**:
    - Increase the depth of the model by adding layers like:
      ```python
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
      ```

16. **Change Pooling Layers**:
    - Replace `MaxPooling2D` with `AveragePooling2D` in some layers to smooth feature maps:
      ```python
      tf.keras.layers.AveragePooling2D(2, 2)
      ```

17. **Adjust Neurons in Dense Layers**:
    - Modify the number of neurons in fully connected layers:
      ```python
      tf.keras.layers.Dense(256, activation='relu')
      ```

18. **Experiment with Activation Functions**:
    - Replace `relu` with other activations like `swish`:
      ```python
      tf.keras.layers.Conv2D(32, (3, 3), activation='swish')
      ```

19. **Change Optimizers**:
    - Test alternative optimizers such as `sgd`, `rmsprop`, or `adamax`:
      ```python
      model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      ```

20. **Add Dropout for Regularization**:
    - Prevent overfitting by adding dropout:
      ```python
      tf.keras.layers.Dropout(0.5)
      ```

21. **Regularization in Dense Layers**:
    - Apply L2 regularization to dense layers:
      ```python
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
      ```

22. **Adjust Training Epochs**:
    - Experiment with more or fewer epochs to avoid underfitting or overfitting.

23. **Change Threshold Accuracy**:
    - Modify the stopping criterion (e.g., `95%` accuracy) to evaluate further gains.

24. **Augment Data During Training**:
    - Add online data augmentation (e.g., rotations, random crops) in the training pipeline.

25. **Adjust Weights and Initializers**:
    - Use `he_normal` or similar weight initializers:
      ```python
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal')
      ```

26. **Fine-Tune Learning Rate**:
    - Use a learning rate scheduler or experiment with different values to stabilize training.

---

#### **Alternative Frameworks (Optional)**
27. **Experiment with Frameworks**:
    - Test your model with different machine learning frameworks for potential performance improvements:
      - **PyTorch**
      - **Keras** (already being used)
      - **TensorFlow Lite** (for edge devices)
      - **ONNX Runtime** (for inference optimization)
      - **Detectron2** or **MMDetection** (for advanced object detection).

---

### **Prioritized To-Do**
1. **Dataset**:
   - Focus on balancing genders, sample size, ethnicity, and augmentations.
   - Standardize image resolutions and incorporate peripheral safety gear.
   - Address age and background diversity.

2. **Preprocessing**:
   - Increase image resolution and switch to color images.
   - Normalize consistently across the dataset.

3. **Training**:
   - Add layers, apply regularization, and fine-tune optimizers and learning rates.
   - Use dropout and augmentation during training to prevent overfitting.

4. **Framework Exploration** (Optional):
   - After stabilizing performance, test the model with frameworks like PyTorch or ONNX Runtime.