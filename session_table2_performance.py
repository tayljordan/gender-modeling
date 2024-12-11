import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Fake data for demonstration purposes
np.random.seed(42)  # Ensures reproducibility

# Generate fake true labels and predictions for a binary classification problem
val_labels = np.random.randint(0, 2, size=1000)  # 1000 samples with labels 0 or 1
val_predictions = np.random.randint(0, 2, size=1000)  # Random predictions (binary)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(val_labels, val_predictions).ravel()

# Calculate performance metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

# Print classification report (optional for verification)
print(classification_report(val_labels, val_predictions, target_names=["Class 0", "Class 1"]))

# Create a DataFrame for Table 2
table2 = pd.DataFrame({
    "Metric": ["True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)",
               "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
    "Value": [tp, tn, fp, fn, accuracy * 100, precision * 100, recall * 100, f1_score * 100]
})

# Display Table 2
print("\nTable 2: Model Performance Metrics")
print(table2)

# Example comparison of multiple models (fake data)
models_metrics = [
    {"Model": "Model A", "TP": 300, "TN": 500, "FP": 100, "FN": 100, "Accuracy (%)": 80.0, "Precision (%)": 75.0, "Recall (%)": 75.0, "F1-Score (%)": 75.0},
    {"Model": "Model B", "TP": 320, "TN": 480, "FP": 120, "FN": 80, "Accuracy (%)": 80.0, "Precision (%)": 72.7, "Recall (%)": 80.0, "F1-Score (%)": 76.2},
    {"Model": "Model C", "TP": 350, "TN": 450, "FP": 150, "FN": 50, "Accuracy (%)": 80.0, "Precision (%)": 70.0, "Recall (%)": 87.5, "F1-Score (%)": 77.8}
]

# Create a DataFrame for all models
comparison_table = pd.DataFrame(models_metrics)

# Display Table 2 for all models
print("\nTable 2: Performance Comparison Across Models")
print(comparison_table)

# Bar plot for F1-Score comparison
comparison_table.plot(
    x='Model',
    y=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
    kind='bar',
    figsize=(10, 6)
)
plt.ylabel('Percentage (%)')
plt.title('Performance Metrics Comparison Across Models')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
