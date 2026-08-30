 #include "LogisticRegressionModel.h"

// Instantiate the micromlgen model
Eloquent::ML::Port::LogisticRegression logReg;

const int NUM_FEATURES = 2;
float inputBuffer[NUM_FEATURES];

void setup() {
    Serial.begin(115200);
    while (!Serial);
}

void loop() {
    if (Serial.available() > 0) {
        // Read the two features: x1, x2
        for (int i = 0; i < NUM_FEATURES; i++) {
            inputBuffer[i] = Serial.parseFloat();
        }

        // Flush trailing newline or carriage return bytes
        while (Serial.available() > 0) {
            Serial.read();
        }

        // Compute predicted class (0 or 1)
        int predicted_class = 1 - logReg.predict(inputBuffer);

        // Send class back to Python
        Serial.println(predicted_class);
    }
}