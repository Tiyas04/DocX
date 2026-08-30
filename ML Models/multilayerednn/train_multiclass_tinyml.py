import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# 1. Load Dataset (4 features, 3 classes)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=30, random_state=42, stratify=y
)

# 2. Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Multi-Class Neural Network (4 -> 8 -> 3)
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 4. Extract Weights, Biases, and Scaler Parameters
W1 = mlp.coefs_[0]        # Shape (4, 8)
b1 = mlp.intercepts_[0]   # Shape (8,)
W2 = mlp.coefs_[1]        # Shape (8, 3)
b2 = mlp.intercepts_[1]   # Shape (3,)
means = scaler.mean_      # Shape (4,)
scales = scaler.scale_    # Shape (4,)

# Helper formatting functions
def format_1d(arr):
    return ", ".join([f"{v:.8f}f" for v in arr])

def format_2d(arr):
    rows = []
    for r in arr:
        rows.append("{" + ", ".join([f"{v:.8f}f" for v in r]) + "}")
    return ",\n        ".join(rows)

# 5. Generate C++ Header for Arduino Uno
cpp_header = f"""#ifndef MULTICLASS_TINYML_MODEL_H
#define MULTICLASS_TINYML_MODEL_H

#include <Arduino.h>
#include <stdint.h>
#include <math.h>

class MultiClassTinyMLModel {{
public:
    static const int INPUT_DIM = 4;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 3;

    // Normalization parameters
    const float means[INPUT_DIM] = {{{format_1d(means)}}};
    const float scales[INPUT_DIM] = {{{format_1d(scales)}}};

    // Layer 1 (Dense: 4 -> 8)
    const float W1[INPUT_DIM][HIDDEN_DIM] = {{
        {format_2d(W1)}
    }};
    const float b1[HIDDEN_DIM] = {{{format_1d(b1)}}};

    // Layer 2 (Dense: 8 -> 3)
    const float W2[HIDDEN_DIM][OUTPUT_DIM] = {{
        {format_2d(W2)}
    }};
    const float b2[OUTPUT_DIM] = {{{format_1d(b2)}}};

    int predict(const float *raw_input) {{
        float normalized_x[INPUT_DIM];
        float hidden[HIDDEN_DIM];
        float output[OUTPUT_DIM];

        // 1. On-device Standard Scaling: (x - mean) / scale
        for (int i = 0; i < INPUT_DIM; i++) {{
            normalized_x[i] = (raw_input[i] - means[i]) / scales[i];
        }}

        // 2. Layer 1: Dense + ReLU
        for (int j = 0; j < HIDDEN_DIM; j++) {{
            float sum = b1[j];
            for (int i = 0; i < INPUT_DIM; i++) {{
                sum += normalized_x[i] * W1[i][j];
            }}
            hidden[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
        }}

        // 3. Layer 2: Output Logits (8 -> 3)
        for (int k = 0; k < OUTPUT_DIM; k++) {{
            float sum = b2[k];
            for (int j = 0; j < HIDDEN_DIM; j++) {{
                sum += hidden[j] * W2[j][k];
            }}
            output[k] = sum;
        }}

        // 4. ArgMax over Logits to determine predicted class
        int best_class = 0;
        float max_logit = output[0];
        for (int k = 1; k < OUTPUT_DIM; k++) {{
            if (output[k] > max_logit) {{
                max_logit = output[k];
                best_class = k;
            }}
        }}

        return best_class;
    }}
}};

static MultiClassTinyMLModel nnClassifier;

#endif
"""

with open("MultiClassTinyMLModel.h", "w") as f:
    f.write(cpp_header)
print("Generated MultiClassTinyMLModel.h successfully!\n")

# 6. Python Software Baseline Predictions
y_py_pred = mlp.predict(X_test_scaled)

# 7. Stream Test Inputs to Arduino Uno on COM5
PORT = 'COM5'
BAUD = 115200

print(f"Connecting to Arduino Uno on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)  # Wait for Arduino bootloader reset

y_arduino_pred = []

for sample in X_test:
    # Send unscaled raw values: "x1,x2,x3,x4\n"
    payload = f"{sample[0]:.3f},{sample[1]:.3f},{sample[2]:.3f},{sample[3]:.3f}\n"
    ser.write(payload.encode('utf-8'))
    
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            y_arduino_pred.append(int(float(line)))
        except ValueError:
            y_arduino_pred.append(-1)
    else:
        y_arduino_pred.append(-1)

ser.close()
y_arduino_pred = np.array(y_arduino_pred)

# 8. Evaluation
matches = np.sum(y_py_pred == y_arduino_pred)
print(f"Software vs Hardware Match: {matches}/{len(y_test)} ({(matches/len(y_test))*100:.1f}%)")

# 9. Comparison Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Prediction Match Comparison
indices = np.arange(1, len(y_test) + 1)
ax1.step(indices, y_py_pred, where='mid', color='blue', linewidth=2.5, label='Python Output (Scikit-Learn)')
ax1.scatter(indices, y_arduino_pred, color='red', marker='o', s=45, label='Arduino Uno TinyML Output')
ax1.set_title('Multi-Class Inference: Python vs Arduino Uno', fontsize=12)
ax1.set_xlabel('Test Sample Index', fontsize=11)
ax1.set_ylabel('Class Label (0: Setosa, 1: Versicolor, 2: Virginica)', fontsize=10)
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Setosa (0)', 'Versicolor (1)', 'Virginica (2)'])
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# Plot 2: Confusion Matrix (Hardware Predictions vs True Labels)
cm = confusion_matrix(y_test, y_arduino_pred)
im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.figure.colorbar(im, ax=ax2)
ax2.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Setosa', 'Versicolor', 'Virginica'],
       yticklabels=['Setosa', 'Versicolor', 'Virginica'],
       title='Arduino Uno Hardware Confusion Matrix',
       ylabel='Ground Truth Label',
       xlabel='Hardware Predicted Label')

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.tight_layout()
plt.show()