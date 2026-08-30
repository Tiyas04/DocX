import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from micromlgen import port

# 1. Generate 2D Binary Classification Dataset
np.random.seed(42)
X_train, y_train = make_classification(
    n_samples=120,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42
)

# 2. Train Logistic Regression Model
clf = LogisticRegression()
clf.fit(X_train, y_train)

w = clf.coef_[0]
b = clf.intercept_[0]

print("--- Trained Model Parameters ---")
print(f"Weight w1:     {w[0]:.6f}")
print(f"Weight w2:     {w[1]:.6f}")
print(f"Intercept b:   {b:.6f}\n")

# 3. Export C++ Code and Patch for AVR / Arduino Uno Compatibility
cpp_code = port(clf)
cpp_code = cpp_code.replace("#include <cstdarg>", "#include <stdarg.h>")
cpp_code = cpp_code.replace("#include <cstdint>", "#include <stdint.h>")
cpp_code = cpp_code.replace("std::uint16_t", "uint16_t")

with open("LogisticRegressionModel.h", "w") as f:
    f.write(cpp_code)

print("Generated LogisticRegressionModel.h successfully!\n")

# 4. Generate Test Points Spanning the Feature Space
N_TEST = 25
np.random.seed(101)
X_test = np.column_stack([
    np.random.uniform(X_train[:, 0].min(), X_train[:, 0].max(), N_TEST),
    np.random.uniform(X_train[:, 1].min(), X_train[:, 1].max(), N_TEST)
])

# Python Ground Truth Predictions (0 or 1)
y_py_pred = clf.predict(X_test)

# 5. Stream Test Points to Arduino Uno on COM5
PORT = 'COM5'
BAUD = 115200

print(f"Connecting to Arduino Uno on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)  # Wait for Arduino bootloader reset

y_arduino_pred = []

for sample in X_test:
    # Send "x1,x2\n"
    payload = f"{sample[0]:.4f},{sample[1]:.4f}\n"
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

# 6. Verification Metrics
matches = np.sum(y_py_pred == y_arduino_pred)
print(f"Accuracy vs Python Baseline: {matches}/{N_TEST} ({(matches/N_TEST)*100:.1f}%)")

# 7. Dual-Panel Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: 2D Feature Space & Decision Boundary
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='#1f77b4', label='Train Class 0', alpha=0.5)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='#d62728', label='Train Class 1', alpha=0.5)

# Plot Decision Boundary Line: w1*x1 + w2*x2 + b = 0
x_vals = np.linspace(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 100)
y_boundary = -(w[0] * x_vals + b) / w[1]
ax1.plot(x_vals, y_boundary, 'k--', linewidth=2, label='Decision Boundary ($z=0$)')

# Overlay Test Points
ax1.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='x', s=50, label='Test Samples')
ax1.set_title('Dataset & Decision Boundary', fontsize=12)
ax1.set_xlabel('Feature 1 ($X_1$)', fontsize=11)
ax1.set_ylabel('Feature 2 ($X_2$)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left')

# Subplot 2: Classification Comparison (Python vs Arduino)
indices = np.arange(1, N_TEST + 1)
ax2.step(indices, y_py_pred, where='mid', color='blue', linewidth=2.5, label='Python Class (Scikit-Learn)')
ax2.scatter(indices, y_arduino_pred, color='red', marker='o', s=50, label='Arduino Uno Class Output')
ax2.set_title('Inference Comparison: Python vs Arduino Uno', fontsize=12)
ax2.set_xlabel('Test Sample Index', fontsize=11)
ax2.set_ylabel('Class Label (0 or 1)', fontsize=11)
ax2.set_yticks([0, 1])
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='center right')

plt.tight_layout()
plt.show()