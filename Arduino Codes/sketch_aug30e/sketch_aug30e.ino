#include "MultiClassTinyMLModel.h"

const int NUM_FEATURES = MultiClassTinyMLModel::INPUT_DIM;
float inputBuffer[NUM_FEATURES];

void setup() {
    Serial.begin(115200);
    while (!Serial);
}

void loop() {
    if (Serial.available() > 0) {
        // Parse 4 comma-separated floating point inputs: x1,x2,x3,x4
        for (int i = 0; i < NUM_FEATURES; i++) {
            inputBuffer[i] = Serial.parseFloat();
        }

        // Flush trailing newline / carriage return bytes
        while (Serial.available() > 0) {
            Serial.read();
        }

        // Run the 2-layer Neural Network forward pass on ATmega328P
        int predicted_class = nnClassifier.predict(inputBuffer);

        // Send hardware predicted class back to PC
        Serial.println(predicted_class);
    }
}