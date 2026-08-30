import numpy as np
from sklearn.linear_model import LinearRegression
from micromlgen import port

# 1. Create synthetic training data: y = 2.5 * x + 3.0 + noise
np.random.seed(42)
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = 2.5 * X_train.squeeze() + 3.0 + np.random.normal(0, 0.2, 50)

# 2. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("--- Model Parameters ---")
print(f"Slope (m):     {model.coef_[0]:.6f}")
print(f"Intercept (c): {model.intercept_:.6f}\n")

# 3. Export C++ code using micromlgen
cpp_code = port(model)

# 4. Make it 100% compatible with Arduino Uno AVR compiler
cpp_code = cpp_code.replace("#include <cstdarg>", "#include <stdarg.h>")
cpp_code = cpp_code.replace("#include <cstdint>", "#include <stdint.h>")
cpp_code = cpp_code.replace("std::uint16_t", "uint16_t")

# 5. Save header file
with open("LinearRegressionModel.h", "w") as f:
    f.write(cpp_code)

print("LinearRegressionModel.h generated successfully!")