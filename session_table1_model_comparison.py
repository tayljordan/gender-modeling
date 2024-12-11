import pandas as pd

# Data for the table
data = {
    'Model': ['Naive Bayes', 'KNN', 'SVM', 'RNN'],
    'Training Accuracy (%)': [80.74, 73.38, 88.86, 95.95],
    'Validation Accuracy (%)': [79.56, 70.12, 85.33, 85.47],
    'Difference (%)': [1.18, 3.26, 3.53, 10.48]  # Example differences
}

# Create a DataFrame
comparison_table = pd.DataFrame(data)

# Display the table
print(comparison_table)

comparison_table.to_csv('model_comparison.csv', index=False)


import matplotlib.pyplot as plt
import pandas as pd

# Example data
data = {
    'Model': ['Naive Bayes', 'KNN', 'SVM', 'RNN'],
    'Training Accuracy (%)': [80.74, 73.38, 88.86, 95.95],
    'Validation Accuracy (%)': [79.56, 70.12, 85.33, 85.47]
}

# Create a DataFrame
comparison_table = pd.DataFrame(data)

# Bar plot for accuracy comparison
fig, ax = plt.subplots(figsize=(8, 5))

comparison_table.plot(
    x='Model',
    y=['Training Accuracy (%)', 'Validation Accuracy (%)'],
    kind='bar',
    ax=ax,
    color=['#1f77b4', '#ff7f0e']  # APA-compliant colors
)

# Add gridlines (APA style)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

# Set labels and title
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Model Comparison: Training vs Validation Accuracy', fontsize=14, pad=15)

# Add legend and format
ax.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left', fontsize=10, frameon=False)

# Adjust ticks
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)

# Add source (if needed for academic context)
plt.figtext(0.5, -0.1, 'Source: Experimental Results', ha='center', fontsize=10)

# Save figure
plt.tight_layout()
plt.savefig('model_comparison_figure.png', dpi=300)  # Save as high-res for academic use
plt.show()
