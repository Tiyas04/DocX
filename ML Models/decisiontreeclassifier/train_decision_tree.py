import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from everywhereml.sklearn.ensemble import RandomForestClassifier

# 1. Non-linear Dataset (Two Interleaved Moons)
np.random.seed(42)
X_train, y_train = make_moons(n_samples=150, noise=0.2, random_state=42)

# 2. Train using everywhereml's native RandomForestClassifier (1 tree = Decision Tree)
clf = RandomForestClassifier(n_estimators=1, max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 3. Export C++ Code using everywhereml's native to_arduino()
cpp_code = clf.to_arduino(instance_name="treeClassifier")

# Patch AVR standard integer/variadic headers if present
cpp_code = cpp_code.replace("#include <cstdarg>", "#include <stdarg.h>")
cpp_code = cpp_code.replace("#include <cstdint>", "#include <stdint.h>")
cpp_code = cpp_code.replace("std::uint16_t", "uint16_t")

with open("DecisionTreeModel.h", "w") as f:
    f.write(cpp_code)

print("DecisionTreeModel.h exported successfully using everywhereml!\n")

# 4. Generate Test Grid across the Moon Feature Space
N_TEST = 25
np.random.seed(101)
X_test = np.column_stack([
    np.random.uniform(X_train[:, 0].min(), X_train[:, 0].max(), N_TEST),
    np.random.uniform(X_train[:, 1].min(), X_train[:, 1].max(), N_TEST)
])

# Python Software Predictions
y_py_pred = clf.predict(X_test)

# 5. Stream Test Inputs to Arduino Uno on COM5
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
print(f"Prediction Match: {matches}/{N_TEST} ({(matches / N_TEST) * 100:.1f}%)")

# 7. Dual Plot: Decision Surface & Hardware Output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: 2D Non-linear Decision Surface
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Train Class 0', alpha=0.6)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', label='Train Class 1', alpha=0.6)
ax1.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='x', s=50, label='Test Points')
ax1.set_title('everywhereml Decision Surface', fontsize=12)
ax1.set_xlabel('Feature 1 ($X_1$)', fontsize=11)
ax1.set_ylabel('Feature 2 ($X_2$)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left')

# Subplot 2: Python vs Hardware Match
indices = np.arange(1, N_TEST + 1)
ax2.step(indices, y_py_pred, where='mid', color='blue', linewidth=2.5, label='Python (everywhereml)')
ax2.scatter(indices, y_arduino_pred, color='red', marker='o', s=50, label='Arduino Uno Hardware Output')
ax2.set_title('Inference Match: Python vs Arduino Uno', fontsize=12)
ax2.set_xlabel('Test Sample Index', fontsize=11)
ax2.set_ylabel('Predicted Class (0 or 1)', fontsize=11)
ax2.set_yticks([0, 1])
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='center right')

plt.tight_layout()
plt.show()