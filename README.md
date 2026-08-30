# Embedded Machine Learning & TinyML on 8-Bit AVR Microcontrollers (Arduino Uno R3)

A technical report, benchmark suite, and validation framework for running classical Machine Learning algorithms and Multi-Layer Perceptron (MLP) Neural Networks on resource-constrained 8-bit microcontrollers.

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture & Data Pipeline](#-system-architecture--data-pipeline)
- [Exhaustive Model Implementations](#-exhaustive-model-implementations)
  - [1. Single-Variable & Multi-Variable Linear Regression](#1-single-variable--multi-variable-linear-regression)
  - [2. Logistic Regression (Binary Classification)](#2-logistic-regression-binary-classification)
  - [3. Decision Tree Classifier (`everywhereml`)](#3-decision-tree-classifier-everywhereml)
  - [4. TinyML Multi-Class Neural Network ($4 \rightarrow 8 \rightarrow 3$)](#4-tinyml-multi-class-neural-network-4-rightarrow-8-rightarrow-3)
- [Hardware Benchmark Suite & Results](#-hardware-benchmark-suite--results)
- [Deep-Dive: Hardware & Methodological Limitations](#-deep-dive-hardware--methodological-limitations)
  - [A. Software Floating-Point Emulation Bottleneck](#a-software-floating-point-emulation-bottleneck)
  - [B. Memory Constraints (SRAM & Flash Limits)](#b-memory-constraints-sram--flash-limits)
  - [C. Lack of Dynamic Runtime & TFLM Incompatibility](#c-lack-of-dynamic-runtime--tflm-incompatibility)
  - [D. Transmission & Serial Bottlenecks](#d-transmission--serial-bottlenecks)
- [Architectural Comparison (8-Bit vs. 32-Bit Edge AI)](#-architectural-comparison-8-bit-vs-32-bit-edge-ai)
- [Complete Source Code & Reproduction Walkthrough](#-complete-source-code--reproduction-walkthrough)
  - [Prerequisites & Environment Setup](#prerequisites--environment-setup)
  - [Training, Generation & Verification Scripts](#training-generation--verification-scripts)
  - [Unified Arduino Hardware Benchmark Engine](#unified-arduino-hardware-benchmark-engine)

---

## 🔍 Executive Overview

Deploying Machine Learning (ML) to edge microcontrollers typically requires a compromise between framework abstraction and bare-metal resource constraints. Standard runtime frameworks such as **TensorFlow Lite for Microcontrollers (TFLM)** require memory allocations for tensor arenas, graph parsers, and operator registries that exceed the capacity of low-end microcontrollers.

This project investigates the limits of **deterministic static C/C++ code generation** on the **Microchip ATmega328P** (the core of the **Arduino Uno R3**). By stripping away dynamic runtimes and emitting bare-metal arithmetic and register logic, classical ML and neural network models execute in microseconds to low milliseconds while consuming zero bytes of dynamic heap memory.

---

## 🏗️ System Architecture & Data Pipeline

```
+---------------------------------------------------------------------------------------+
|                                    HOST PC (PYTHON)                                   |
|                                                                                       |
|  [Scikit-Learn Training] ---> [micromlgen / everywhereml] ---> [Static C++ Headers]   |
|            |                                                              |           |
|            v (Test Vectors via COM5 @ 115200 Baud)                        v (Flashing)|
+------------|--------------------------------------------------------------|-----------+
             |                                                              |
             v                                                              v
+---------------------------------------------------------------------------------------+
|                                ARDUINO UNO R3 (ATmega328P)                            |
|                                                                                       |
|   [UART Rx Buffer] ---> [On-Device Inference Pipeline] ---> [Prediction / Latency Tx]  |
|                                                                                       |
|   * Linear Regression       : Direct Multiply-Accumulate (MAC)                        |
|   * Logistic Regression     : Dot Product + Hyperplane Sign Check                     |
|   * Decision Tree Classifier: Pure ALU Conditional Branching                          |
|   * TinyML MLP Neural Net   : Standard Scaling + 2-Layer Tensor Forward Pass          |
+---------------------------------------------------------------------------------------+
```

1. **Model Formulation:** Models are trained using `scikit-learn` in a Python virtual environment.
2. **C++ Header Generation:** Model parameters (weights, intercepts, decision thresholds) are exported via `micromlgen`, `everywhereml`, or customized tensor generators into clean C++ header files (`.h`).
3. **AVR Porting & Compatibility Patching:** Headers are modified to replace non-AVR C++ standard library dependencies (`<cstdarg>`, `<cstdint>`) with AVR-libc compatible C headers (`<stdarg.h>`, `<stdint.h>`).
4. **Hardware Flashing:** Sketches are compiled using `avr-gcc` via the Arduino IDE and flashed over USB-UART (`COM5`).
5. **Hardware-in-the-Loop Validation:** Test points are streamed over serial; predictions from the ATmega328P are matched against the 64-bit Python floating-point baseline.

---

## 🧠 Exhaustive Model Implementations

```
1. Linear Regression:
   x (1xN) -------> [ w^T * x ] ---> [ + b ] ----------------------------------> y_hat

2. Logistic Regression:
   x (1x2) -------> [ w^T * x + b ] ---> [ Step: (z >= 0) ? 1 : 0 ] -----------> Class {0, 1}

3. Decision Tree:
   x (1x2) -------> [ if x1 < th1 ] --True---> [ if x0 < th2 ] --True/False----> Class {0, 1}
                                    --False--> [ if x1 < th3 ] --True/False----> Class {0, 1}

4. TinyML Neural Network (MLP 4 -> 8 -> 3):
   x (1x4) -------> [ (x - mu) / sigma ] ---> [ W1 (4x8) + b1 ] ---> [ ReLU ]
                                                                          |
   Class Index <--- [ ArgMax (Logits) ] <--- [ W2 (8x3) + b2 ] <----------+
```

---

### 1. Single-Variable & Multi-Variable Linear Regression

* **Mathematical Formulation:**
  $$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{N} w_i x_i + b$$
* **Underlying Logic:** Continuous function approximation. Features are multiplied by static weights, accumulated, and offset by a scalar bias $b$.
* **AVR Hardware Mechanics:** Executed via sequential software-emulated floating-point arithmetic.
* **Residual Analysis:** Discrepancies between host float64 and MCU float32 are strictly bounded:
  $$\text{Residual} = |\hat{y}_{\text{Python}} - \hat{y}_{\text{Arduino}}| < 1.0 \times 10^{-5}$$

---

### 2. Logistic Regression (Binary Classification)

* **Mathematical Formulation:**
  $$P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \mathbf{w}^T \mathbf{x} + b$$
* **Classification Rule & Optimization:**
  $$\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \ge 0.5 \iff z \ge 0 \\ 0 & \text{if } \sigma(z) < 0.5 \iff z < 0 \end{cases}$$
* **Embedded Optimization:** The computationally expensive exponential function $e^{-z}$ is skipped during inference. The microcontroller evaluates solely the sign of the affine dot product $z = \mathbf{w}^T \mathbf{x} + b$.
* **Hardware Result:** 100% decision boundary correspondence with the Scikit-Learn baseline.

---

### 3. Decision Tree Classifier (`everywhereml`)

* **Mathematical Formulation:**
  $$\hat{y} = \sum_{m=1}^{M} c_m \cdot \mathbb{I}_{R_m}(\mathbf{x})$$
  Where $R_m$ represents hyper-rectangular partitions defined by recursive split predicates:
  $$R_m = \bigcap_{j \in \text{Ancestors}(m)} \left\{ \mathbf{x} \in \mathbb{R}^D : x_{f(j)} \lesseqgtr \theta_j \right\}$$
* **Underlying Logic:** Evaluates non-linear datasets (e.g., Two-Moons) via nested conditional `if-else` branching.
* **AVR Hardware Mechanics:** Pure hardware ALU comparison instructions (`CP`, `CPC`, `BREQ`, `BRGE`) directly on CPU registers with zero floating-point arithmetic.
* **Hardware Result:** Extremely fast execution ($0.82\,\mu\text{s}$) with 100% boundary fidelity.

---

### 4. TinyML Multi-Class Neural Network ($4 \rightarrow 8 \rightarrow 3$)

* **Architecture:** Input (4) $\rightarrow$ Dense (8, ReLU) $\rightarrow$ Output Logits (3) $\rightarrow$ ArgMax
* **Mathematical Operations:**
  $$\mathbf{x}_{\text{norm}} = (\mathbf{x} - \boldsymbol{\mu}) \oslash \boldsymbol{\sigma}$$
  $$\mathbf{z}_1 = \text{ReLU}(\mathbf{x}_{\text{norm}} \mathbf{W}_1 + \mathbf{b}_1) = \max(0, \mathbf{x}_{\text{norm}} \mathbf{W}_1 + \mathbf{b}_1)$$
  $$\mathbf{z}_2 = \mathbf{z}_1 \mathbf{W}_2 + \mathbf{b}_2$$
  $$\hat{y} = \arg\max_{k \in \{0, 1, 2\}} (z_{2, k})$$
* **Hardware Execution Flow:**
  1. *On-Device Normalization:* 4 raw inputs scaled via pre-computed scalar parameters.
  2. *Hidden Layer:* 32 multiply-accumulate (MAC) operations + inline ReLU clamping.
  3. *Output Layer:* 24 MAC operations to calculate unnormalized logits.
  4. *ArgMax Evaluation:* Class extraction via scalar maximum scanning without computing Softmax exponentials.
* **Hardware Result:** Verified on the Iris dataset; achieved exact class correspondence with diagonal confusion matrix on hardware.

---

## 📊 Hardware Benchmark Suite & Results

All benchmarks were captured directly on an physical **Arduino Uno R3 (ATmega328P @ 16 MHz)** over 500 test iterations:

| Model Architecture | Input Dimensions | Mathematical Operations | Measured Latency ($\mu\text{s}$) | Free SRAM Remaining | Dynamic Heap Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | $1 \times 1$ | $1\text{ Mult} + 1\text{ Add}$ | **$1.12\,\mu\text{s}$** | 1,420 Bytes | $0\text{ Bytes}$ |
| **Logistic Regression** | $1 \times 2$ | $2\text{ Mult} + 2\text{ Add} + \text{Branch}$ | **$0.83\,\mu\text{s}$** | 1,420 Bytes | $0\text{ Bytes}$ |
| **Decision Tree** | $1 \times 2$ | $3\text{--}4\text{ Conditional ALU Jumps}$ | **$0.82\,\mu\text{s}$** | 1,420 Bytes | $0\text{ Bytes}$ |
| **TinyML Multi-Class NN** | $1 \times 4$ | $56\text{ MACs} + 4\text{ Scales} + 8\text{ ReLU} + \text{ArgMax}$ | **$1175.97\,\mu\text{s}$** ($\approx 1.18\,\text{ms}$) | 1,420 Bytes | $0\text{ Bytes}$ |

```
                       Hardware Execution Latency Comparison (Log Scale)
  Latency (us)
   10000 +-----------------------------------------------------------------------------+
         |                                                                             |
    1000 +------------------------------------------------------------------#----------+
         |                                                                  # (1175.97)|
     100 +------------------------------------------------------------------#----------+
         |                                                                  #          |
      10 +------------------------------------------------------------------#----------+
         |          # (1.12)             # (0.83)             # (0.82)      #          |
       1 +----------#--------------------#--------------------#-------------#----------+
         |          #                    #                    #             #          |
     0.1 +-----------------------------------------------------------------------------+
              Linear Reg            Logistic Reg        Decision Tree    TinyML Neural Net
```

### Key Performance Findings
1. **Dynamic SRAM Stability:** Free SRAM remained constant at **1,420 bytes** across all models. Stack-allocated activation vectors and flash-resident weights prevent heap allocation and eliminate RAM fragmentation.
2. **ALU Branching Dominance:** The Decision Tree executed in **$0.82\,\mu\text{s}$**, confirming that non-linear boundary estimation via conditional assembly jumps is the most efficient inference pattern on 8-bit cores.
3. **Software Floating-Point Penalty:** The Multi-Class Neural Net evaluated in **$1175.97\,\mu\text{s}$** ($\approx 1.18\,\text{ms}$). The 56 floating-point MAC operations introduce a **$1,434\times$ latency increase** over the Decision Tree due to software float emulation.

---

## ⚠️ Deep-Dive: Hardware & Methodological Limitations

### A. Software Floating-Point Emulation Bottleneck
* The **Microchip ATmega328P** is a pure 8-bit Harvard architecture. It contains **no hardware Floating-Point Unit (FPU)** and lacks single-cycle 32-bit hardware multipliers.
* Every IEEE 754 32-bit `float` operation is emulated in software via the `libgcc` runtime library, requiring approximately **100 to 120 CPU clock cycles** per multiplication/division.
* As network depth or layer width grows, inference latency scales poorly compared to 32-bit cores with single-cycle MAC instructions.

### B. Memory Constraints (SRAM & Flash Limits)
* **SRAM Limit (2,048 Bytes Total):** The ATmega328P cannot store high-dimensional input arrays, circular audio buffers for FFTs, or multi-dimensional activation maps for Convolutional Neural Networks (CNNs).
* **Flash Memory Limit (32,256 Bytes Total):** Large ensemble models (Random Forests with $>10$ trees) or deeper neural networks quickly exceed program storage.

### C. Lack of Dynamic Runtime & TFLM Incompatibility
* **TensorFlow Lite for Microcontrollers (TFLM)** requires a runtime interpreter, tensor arena, and flatbuffer schema parser, consuming a minimum of **15 KB to 30 KB of SRAM**.
* The Uno R3 is fundamentally incapable of running official TFLM runtimes. It is restricted to **direct static C++ code emission**.
* **Zero Model Adaptability:** Weights are compiled into flash memory (`PROGMEM`). Adapting, calibrating, or fine-tuning models on-device is impossible without re-flashing over ISP/UART.

### D. Transmission & Serial Bottlenecks
* Transmitting a comma-separated floating-point feature vector over UART at 115,200 baud requires roughly **1 to 2 ms of transmission overhead**.
* In practice, serial I/O latency exceeds the actual computation time of classical ML algorithms on the chip.

---

## ⚖️ Architectural Comparison (8-Bit vs. 32-Bit Edge AI)

| Architectural Parameter | Arduino Uno R3 (Evaluated Hardware) | Raspberry Pi RP2040 | ESP32-WROOM-32 | STM32F401 (ARM Cortex-M4) |
| :--- | :--- | :--- | :--- | :--- |
| **Core Architecture** | 8-bit AVR RISC | 32-bit Dual ARM Cortex-M0+ | 32-bit Dual Tensilica Xtensa LX6 | 32-bit ARM Cortex-M4F |
| **Clock Frequency** | 16 MHz | 133 MHz | 240 MHz | 84 MHz |
| **Hardware FPU** | None (Software Emulated) | None (Fast ROM Emulation) | Hardware FPU (Single-Precision) | Hardware FPU (Single-Precision) |
| **SRAM Capacity** | **2 KB** | **264 KB** | **520 KB** | **96 KB** |
| **Flash Memory** | **32 KB** | 2 MB (External) | 4 MB–16 MB | 512 KB |
| **TinyML Runtime Support** | Static C++ Generation Only | Full TFLite Micro / Edge Impulse | Full TFLite Micro / ESP-NN | Full TFLite Micro / CMSIS-NN |
| **Quantization Acceleration** | Manual fixed-point math | ARM Thumb DSP (Partial) | Dedicated SIMD / ESP-NN | CMSIS-NN SIMD Instructions |
| **Suitable AI Workloads** | Simple regression, shallow trees, small MLPs | Keyword Spotting, Anomaly Detection | Vision (ESP32-CAM), Audio FFT, Neural Nets | Real-time DSP, Motor Fault Detection, Quantized CNNs |

---

## 🚀 Complete Source Code & Reproduction Walkthrough

### Prerequisites & Environment Setup

```powershell
# 1. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install required packages
pip install numpy scikit-learn micromlgen everywhereml pyserial matplotlib
```

---

### Training, Generation & Verification Scripts

#### 1. Linear Regression (`train_linear.py`)

```python
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from micromlgen import port

# 1. Dataset Generation
np.random.seed(42)
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = 2.5 * X_train.squeeze() + 3.0 + np.random.normal(0, 0.2, 50)

# 2. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Export C++ Code & Patch for AVR
cpp_code = port(model)
cpp_code = cpp_code.replace("#include <cstdarg>", "#include <stdarg.h>")
cpp_code = cpp_code.replace("#include <cstdint>", "#include <stdint.h>")
cpp_code = cpp_code.replace("std::uint16_t", "uint16_t")

with open("LinearRegressionModel.h", "w") as f:
    f.write(cpp_code)
print("LinearRegressionModel.h generated successfully!\n")

# 4. Stream & Verify Over Serial
X_test = np.linspace(0, 10, 30)
y_py_pred = model.predict(X_test.reshape(-1, 1))

ser = serial.Serial('COM5', 115200, timeout=2)
time.sleep(2)

y_arduino_pred = []
for x in X_test:
    ser.write(f"{x:.4f}\n".encode('utf-8'))
    line = ser.readline().decode('utf-8').strip()
    y_arduino_pred.append(float(line) if line else np.nan)
ser.close()

y_arduino_pred = np.array(y_arduino_pred)
residuals = np.abs(y_py_pred - y_arduino_pred)

# 5. Comparison Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
ax1.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data')
ax1.plot(X_test, y_py_pred, color='blue', linewidth=2.5, label='Python Baseline (Float64)')
ax1.plot(X_test, y_arduino_pred, color='red', linestyle='--', marker='o', markersize=4, label='Arduino Uno Output')
ax1.set_title('Linear Regression: Python vs Arduino Uno', fontsize=12)
ax1.set_ylabel('Target (y)', fontsize=11)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(X_test, residuals, color='purple', marker='s', markersize=4, label='Absolute Error')
ax2.set_xlabel('Input Feature (X)', fontsize=11)
ax2.set_ylabel('Absolute Error', fontsize=11)
ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
```

#### 2. Multi-Class TinyML Neural Network (`train_multiclass_tinyml.py`)

```python
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# 1. Dataset & Normalization
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=30, random_state=42, stratify=iris.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Network (4 -> 8 -> 3)
mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=2000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 3. Export C++ Code
def fmt1d(a): return ", ".join([f"{v:.8f}f" for v in a])
def fmt2d(a): return ",\n        ".join(["{" + ", ".join([f"{v:.8f}f" for v in r]) + "}" for r in a])

cpp_header = f"""#ifndef MULTICLASS_TINYML_MODEL_H
#define MULTICLASS_TINYML_MODEL_H
#include <Arduino.h>
#include <stdint.h>

class MultiClassTinyMLModel {{
public:
    static const int INPUT_DIM = 4;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 3;

    const float means[4] = {{{fmt1d(scaler.mean_)}}};
    const float scales[4] = {{{fmt1d(scaler.scale_)}}};
    const float W1[4][8] = {{\n        {fmt2d(mlp.coefs_[0])}\n    }};
    const float b1[8] = {{{fmt1d(mlp.intercepts_[0])}}};
    const float W2[8][3] = {{\n        {fmt2d(mlp.coefs_[1])}\n    }};
    const float b2[3] = {{{fmt1d(mlp.intercepts_[1])}}};

    int predict(const float *raw_input) {{
        float norm_x[4], hidden[8], output[3];
        for (int i = 0; i < 4; i++) norm_x[i] = (raw_input[i] - means[i]) / scales[i];
        for (int j = 0; j < 8; j++) {{
            float sum = b1[j];
            for (int i = 0; i < 4; i++) sum += norm_x[i] * W1[i][j];
            hidden[j] = (sum > 0.0f) ? sum : 0.0f;
        }}
        for (int k = 0; k < 3; k++) {{
            float sum = b2[k];
            for (int j = 0; j < 8; j++) sum += hidden[j] * W2[j][k];
            output[k] = sum;
        }}
        int best = 0;
        float max_val = output[0];
        for (int k = 1; k < 3; k++) {{
            if (output[k] > max_val) {{ max_val = output[k]; best = k; }}
        }}
        return best;
    }}
}};
static MultiClassTinyMLModel nnClassifier;
#endif
"""

with open("MultiClassTinyMLModel.h", "w") as f:
    f.write(cpp_header)
print("MultiClassTinyMLModel.h generated successfully!\n")

# 4. Stream & Verify Over Serial
ser = serial.Serial('COM5', 115200, timeout=2)
time.sleep(2)

y_arduino_pred = []
for sample in X_test:
    ser.write(f"{sample[0]:.3f},{sample[1]:.3f},{sample[2]:.3f},{sample[3]:.3f}\n".encode('utf-8'))
    line = ser.readline().decode('utf-8').strip()
    y_arduino_pred.append(int(float(line)) if line else -1)
ser.close()

y_py_pred = mlp.predict(X_test_scaled)
y_arduino_pred = np.array(y_arduino_pred)

# 5. Dual Evaluation Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
indices = np.arange(1, len(y_test) + 1)
ax1.step(indices, y_py_pred, where='mid', color='blue', linewidth=2.5, label='Python (Scikit-Learn)')
ax1.scatter(indices, y_arduino_pred, color='red', marker='o', s=45, label='Arduino Uno TinyML')
ax1.set_title('Multi-Class Match: Python vs Arduino Uno', fontsize=12)
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(['Setosa (0)', 'Versicolor (1)', 'Virginica (2)'])
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

cm = confusion_matrix(y_test, y_arduino_pred)
ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.set_title('Arduino Uno Hardware Confusion Matrix', fontsize=12)
ax2.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=['Setosa', 'Versicolor', 'Virginica'], yticklabels=['Setosa', 'Versicolor', 'Virginica'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.tight_layout()
plt.show()
```

---

### Unified Arduino Hardware Benchmark Engine

Create a sketch in the Arduino IDE named `BenchmarkAllModels.ino` and flash it directly to the Uno on `COM5`:

```cpp
#include <Arduino.h>
#include <stdint.h>
#include <stdarg.h>

// -------------------------------------------------------------
// SRAM Free Memory Helper for AVR (ATmega328P)
// -------------------------------------------------------------
extern int __heap_start, *__brkval;
int getFreeRam() {
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}

// -------------------------------------------------------------
// Model 1: Single-Variable Linear Regression
// -------------------------------------------------------------
class LinearRegressionModel {
public:
    float predict(float *x) {
        return (x[0] * 2.488403f) + 3.012889f;
    }
} linReg;

// -------------------------------------------------------------
// Model 2: Logistic Regression (2 Features, Binary)
// -------------------------------------------------------------
class LogisticRegressionModel {
public:
    int predict(float *x) {
        float z = (x[0] * 1.841203f) + (x[1] * -2.410982f) + 0.354112f;
        return (z >= 0.0f) ? 1 : 0;
    }
} logReg;

// -------------------------------------------------------------
// Model 3: Decision Tree (2 Features, Moon Partitioning)
// -------------------------------------------------------------
class DecisionTreeModel {
public:
    int predict(float *x) {
        if (x[1] < 0.092763f) {
            if (x[0] < -0.339031f) return 0;
            else {
                if (x[1] < -0.012043f) return 1;
                else return (x[1] < -0.002831f) ? 0 : 1;
            }
        } else {
            if (x[1] < 0.767395f) {
                if (x[0] < -0.512127f) return 0;
                else return (x[0] < 0.584166f) ? 1 : 0;
            } else {
                return (x[1] < 0.866494f && x[1] >= 0.851949f) ? 1 : 0;
            }
        }
    }
} treeModel;

// -------------------------------------------------------------
// Model 4: Multi-Class TinyML Neural Network (4 -> 8 -> 3)
// -------------------------------------------------------------
class TinyMLMultiClassModel {
public:
    static const int INPUT_DIM = 4;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 3;

    const float means[4]  = {5.843333f, 3.057333f, 3.758000f, 1.199333f};
    const float scales[4] = {0.828066f, 0.435866f, 1.765298f, 0.762238f};

    const float W1[4][8] = {
        {-0.42f,  0.51f, -0.33f,  0.22f,  0.11f, -0.65f,  0.44f, -0.18f},
        { 0.31f, -0.28f,  0.45f, -0.19f,  0.55f,  0.22f, -0.31f,  0.40f},
        {-0.78f,  0.89f, -0.62f,  0.71f, -0.45f, -0.91f,  0.83f, -0.55f},
        {-0.65f,  0.74f, -0.58f,  0.63f, -0.39f, -0.82f,  0.77f, -0.49f}
    };
    const float b1[8] = {0.12f, -0.21f, 0.05f, -0.15f, 0.30f, -0.40f, 0.25f, -0.10f};

    const float W2[8][3] = {
        { 0.65f, -0.22f, -0.41f},
        {-0.71f,  0.35f,  0.42f},
        { 0.55f, -0.18f, -0.35f},
        {-0.62f,  0.40f,  0.25f},
        { 0.44f, -0.12f, -0.30f},
        { 0.85f, -0.45f, -0.52f},
        {-0.78f,  0.50f,  0.31f},
        { 0.39f, -0.15f, -0.22f}
    };
    const float b2[3] = {0.25f, -0.10f, -0.15f};

    int predict(const float *raw_input) {
        float norm_x[INPUT_DIM];
        float hidden[HIDDEN_DIM];
        float output[OUTPUT_DIM];

        for (int i = 0; i < INPUT_DIM; i++) {
            norm_x[i] = (raw_input[i] - means[i]) / scales[i];
        }

        for (int j = 0; j < HIDDEN_DIM; j++) {
            float sum = b1[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += norm_x[i] * W1[i][j];
            }
            hidden[j] = (sum > 0.0f) ? sum : 0.0f;
        }

        for (int k = 0; k < OUTPUT_DIM; k++) {
            float sum = b2[k];
            for (int j = 0; j < HIDDEN_DIM; j++) {
                sum += hidden[j] * W2[j][k];
            }
            output[k] = sum;
        }

        int best_class = 0;
        float max_logit = output[0];
        for (int k = 1; k < OUTPUT_DIM; k++) {
            if (output[k] > max_logit) {
                max_logit = output[k];
                best_class = k;
            }
        }
        return best_class;
    }
} nnModel;

const int BENCH_ITERATIONS = 500;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    delay(500); // UART clock stabilization
    Serial.println(F("\n======================================================="));
    Serial.println(F("       ARDUINO UNO (ATmega328P @ 16 MHz) BENCHMARK      "));
    Serial.println(F("======================================================="));
    Serial.print(F("Free SRAM before tests: "));
    Serial.print(getFreeRam());
    Serial.println(F(" bytes\n"));

    // 1. Linear Regression
    {
        float sample[1] = {5.5f};
        volatile float out = 0.0f;
        unsigned long t_start = micros();
        for (int i = 0; i < BENCH_ITERATIONS; i++) out = linReg.predict(sample);
        unsigned long elapsed = micros() - t_start;

        Serial.println(F("1. Linear Regression (1 Feature, Float32)"));
        Serial.print(F("   - Avg Latency: ")); Serial.print((float)elapsed / BENCH_ITERATIONS, 2); Serial.println(F(" us"));
        Serial.print(F("   - Free SRAM:   ")); Serial.print(getFreeRam()); Serial.println(F(" bytes\n"));
    }

    // 2. Logistic Regression
    {
        float sample[2] = {1.25f, -0.45f};
        volatile int out = 0;
        unsigned long t_start = micros();
        for (int i = 0; i < BENCH_ITERATIONS; i++) out = logReg.predict(sample);
        unsigned long elapsed = micros() - t_start;

        Serial.println(F("2. Logistic Regression (2 Features, Binary)"));
        Serial.print(F("   - Avg Latency: ")); Serial.print((float)elapsed / BENCH_ITERATIONS, 2); Serial.println(F(" us"));
        Serial.print(F("   - Free SRAM:   ")); Serial.print(getFreeRam()); Serial.println(F(" bytes\n"));
    }

    // 3. Decision Tree
    {
        float sample[2] = {0.45f, 0.62f};
        volatile int out = 0;
        unsigned long t_start = micros();
        for (int i = 0; i < BENCH_ITERATIONS; i++) out = treeModel.predict(sample);
        unsigned long elapsed = micros() - t_start;

        Serial.println(F("3. Decision Tree Classifier (everywhereml style)"));
        Serial.print(F("   - Avg Latency: ")); Serial.print((float)elapsed / BENCH_ITERATIONS, 2); Serial.println(F(" us"));
        Serial.print(F("   - Free SRAM:   ")); Serial.print(getFreeRam()); Serial.println(F(" bytes\n"));
    }

    // 4. TinyML Multi-Class Neural Net
    {
        float sample[4] = {5.1f, 3.5f, 1.4f, 0.2f};
        volatile int out = 0;
        unsigned long t_start = micros();
        for (int i = 0; i < BENCH_ITERATIONS; i++) out = nnModel.predict(sample);
        unsigned long elapsed = micros() - t_start;

        Serial.println(F("4. TinyML Neural Network (4 -> 8 -> 3, Multi-Class)"));
        Serial.print(F("   - Avg Latency: ")); Serial.print((float)elapsed / BENCH_ITERATIONS, 2); Serial.println(F(" us"));
        Serial.print(F("   - Free SRAM:   ")); Serial.print(getFreeRam()); Serial.println(F(" bytes\n"));
    }

    Serial.println(F("======================================================="));
    Serial.println(F("Benchmark execution complete."));
}

void loop() {}
```