#ifndef TINYML_NEURAL_NET_H
#define TINYML_NEURAL_NET_H

#include <stdint.h>
#include <Arduino.h>

class TinyMLModel {
public:
    static const int INPUT_DIM = 1;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 1;

    const float W1[8] = {-0.23587228f, 0.80436988f, 0.50069015f, 0.10842483f, -0.21278907f, -0.25881929f, -0.40250526f, 0.56491137f};
    const float b1[8] = {0.21831893f, 0.57759370f, -0.84398536f, 0.61847530f, 0.94481670f, -0.46974827f, -0.51957765f, -0.68487765f};
    const float W2[8] = {-0.45234431f, 0.12907078f, -0.28294806f, -0.20818948f, 0.63311378f, -0.28241596f, -0.08336567f, -0.26237971f};
    const float b2 = 0.10119527f;

    float predict(float *x) {
        float hidden[HIDDEN_DIM];

        // Layer 1: Dense + ReLU
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float sum = (x[0] * W1[i]) + b1[i];
            hidden[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU Activation
        }

        // Layer 2: Output Linear
        float output = b2;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            output += hidden[i] * W2[i];
        }

        return output;
    }
};

static TinyMLModel tinyml;

#endif
