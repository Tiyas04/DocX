#include "DecisionTreeModel.h"

// treeClassifier is instantiated by everywhereml
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

        // Clear leftover newline bytes
        while (Serial.available() > 0) {
            Serial.read();
        }

        // Run inference on the Uno's AVR core
        int predicted_class = treeClassifier.predict(inputBuffer);

        // Send predicted class back to PC
        Serial.println(predicted_class);
    }
}