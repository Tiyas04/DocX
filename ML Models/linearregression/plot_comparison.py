import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Dataset & Software Ground Truth
np.random.seed(42)
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = 2.5 * X_train.squeeze() + 3.0 + np.random.normal(0, 0.2, 50)

model = LinearRegression()
model.fit(X_train, y_train)

# 2. Test values to send to Arduino
X_test = np.linspace(0, 10, 30)
y_py_pred = model.predict(X_test.reshape(-1, 1))

# 3. Stream inputs to Arduino Uno on COM5
PORT = 'COM5'
BAUD = 115200

print(f"Connecting to Arduino Uno on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)  # Wait for Arduino to reset after connection

y_arduino_pred = []

for x in X_test:
    # Send test value to Arduino
    ser.write(f"{x:.4f}\n".encode('utf-8'))
    
    # Read response back from Arduino
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

# 4. Compute Residual Error between Python and Hardware
residuals = np.abs(y_py_pred - y_arduino_pred)

# 5. Display Dual Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

# Top: Model Output Comparison
ax1.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data (Noisy)')
ax1.plot(X_test, y_py_pred, color='blue', linewidth=2.5, label='Python Baseline (Float64)')
ax1.plot(X_test, y_arduino_pred, color='red', linestyle='--', marker='o', markersize=4, label='Arduino Uno Output (micromlgen C++)')
ax1.set_title('Linear Regression: Software (Python) vs Hardware (Arduino Uno)', fontsize=12, pad=10)
ax1.set_ylabel('Target (y)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Bottom: Precision Residuals
ax2.plot(X_test, residuals, color='purple', marker='s', markersize=4, label='Absolute Error |Py - Uno|')
ax2.set_title('Hardware Floating-Point Discrepancy', fontsize=10)
ax2.set_xlabel('Input Feature (X)', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
y_max = np.nanmax(residuals) if not np.all(np.isnan(residuals)) else 1e-5
ax2.set_ylim(-1e-6, max(y_max * 1.5, 1e-5))
ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()