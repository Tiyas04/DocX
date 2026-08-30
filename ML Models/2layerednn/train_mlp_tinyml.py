import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 1. Generate Non-Linear Sine Wave Dataset
np.random.seed(42)
X_train = np.linspace(0, 2 * np.pi, 60).reshape(-1, 1)
y_train = np.sin(X_train.squeeze()) + np.random.normal(0, 0.05, 60)

# 2. Train a 2-Layer Neural Network (MLP: 1 Input -> 8 Hidden Neurons -> 1 Output)
mlp = MLPRegressor(
    hidden_layer_sizes=(8,),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=42
)
mlp.fit(X_train, y_train)

# Extract Weights and Biases
W1 = mlp.coefs_[0]       # Shape (1, 8)
b1 = mlp.intercepts_[0]  # Shape (8,)
W2 = mlp.coefs_[1]       # Shape (8, 1)
b2 = mlp.intercepts_[1]  # Shape (1,)

print("--- Neural Network Architecture ---")
print(f"Layer 1 (Dense): Input (1) -> Hidden (8) with ReLU")
print(f"Layer 2 (Dense): Hidden (8) -> Output (1) Linear\n")

# 3. Export C++ TinyML Tensor Engine
w1_str = ", ".join([f"{val:.8f}f" for val in W1.flatten()])
b1_str = ", ".join([f"{val:.8f}f" for val in b1.flatten()])
w2_str = ", ".join([f"{val:.8f}f" for val in W2.flatten()])
b2_str = f"{b2[0]:.8f}f"

cpp_code = f"""#ifndef TINYML_NEURAL_NET_H
#define TINYML_NEURAL_NET_H

#include <stdint.h>
#include <Arduino.h>

class TinyMLModel {{
public:
    static const int INPUT_DIM = 1;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 1;

    const float W1[8] = {{{w1_str}}};
    const float b1[8] = {{{b1_str}}};
    const float W2[8] = {{{w2_str}}};
    const float b2 = {b2_str};

    float predict(float *x) {{
        float hidden[HIDDEN_DIM];

        // Layer 1: Dense + ReLU
        for (int i = 0; i < HIDDEN_DIM; i++) {{
            float sum = (x[0] * W1[i]) + b1[i];
            hidden[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU Activation
        }}

        // Layer 2: Output Linear
        float output = b2;
        for (int i = 0; i < HIDDEN_DIM; i++) {{
            output += hidden[i] * W2[i];
        }}

        return output;
    }}
}};

static TinyMLModel tinyml;

#endif
"""

with open("TinyMLModel.h", "w") as f:
    f.write(cpp_code)

print("TinyMLModel.h generated successfully!\n")

# 4. Generate Test Grid for Evaluation
N_TEST = 30
X_test = np.linspace(0, 2 * np.pi, N_TEST).reshape(-1, 1)
y_py_pred = mlp.predict(X_test)

# 5. Stream Test Inputs to Arduino Uno on COM5
PORT = 'COM5'
BAUD = 115200

print(f"Connecting to Arduino Uno on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)  # Wait for Arduino bootloader reset

y_arduino_pred = []

for sample in X_test:
    ser.write(f"{sample[0]:.4f}\n".encode('utf-8'))
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            y_arduino_pred.append(float(line))
        except ValueError:
            y_arduino_pred.append(np.nan)
    else:
        y_arduino_pred.append(np.nan)

ser.close()
y_arduino_pred = np.array(y_arduino_pred)

# 6. Compute Discrepancy (Residuals)
residuals = np.abs(y_py_pred - y_arduino_pred)

# 7. Dual-Panel Comparison Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

# Top Plot: Sine Wave Regression
ax1.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data ($\sin(x) + \epsilon$)')
ax1.plot(X_test, y_py_pred, color='blue', linewidth=2.5, label='Python Neural Net (Scikit-Learn)')
ax1.plot(X_test, y_arduino_pred, color='red', linestyle='--', marker='o', markersize=5, label='Arduino Uno TinyML Hardware Output')
ax1.set_title('TinyML Neural Network ($\sin(x)$ Non-Linear Regression): Python vs Arduino Uno', fontsize=12, pad=10)
ax1.set_ylabel('Output Target ($y$)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# Bottom Plot: Residuals
ax2.plot(X_test, residuals, color='purple', marker='s', markersize=4, label='Precision Error |Py - Uno|')
ax2.set_title('Hardware Floating-Point Discrepancy', fontsize=10)
ax2.set_xlabel('Input Radians ($x$)', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
y_max = np.nanmax(residuals) if not np.all(np.isnan(residuals)) else 1e-5
ax2.set_ylim(-1e-6, max(y_max * 1.5, 1e-5))
ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()