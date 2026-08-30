#include "TinyMLModel.h"

const int NUM_INPUTS = 1;
float inputBuffer[NUM_INPUTS];

void setup() {
    Serial.begin(115200);
    while (!Serial);
}

void loop() {
    if (Serial.available() > 0) {
        inputBuffer[0] = Serial.parseFloat();

        // Clear trailing newline bytes
        while (Serial.available() > 0) {
            Serial.read();
        }

        // Run forward pass of the 2-layer Neural Network on ATmega328P
        float prediction = tinyml.predict(inputBuffer);

        // Send hardware prediction back to PC
        Serial.println(prediction, 6);
    }
}