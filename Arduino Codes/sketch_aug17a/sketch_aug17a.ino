#include "LinearRegressionModel.h"

void setup() {
    Serial.begin(115200);
    while (!Serial);
}

void loop() {
    if (Serial.available() > 0) {
        float x_val = Serial.parseFloat();
        
        // Clear any leftover newline or carriage return characters
        while (Serial.available() > 0) {
            Serial.read();
        }

        float input[1] = {x_val};
        float y_pred = linReg.predict(input);

        // Send hardware prediction back to Python
        Serial.println(y_pred, 6);
    }
}