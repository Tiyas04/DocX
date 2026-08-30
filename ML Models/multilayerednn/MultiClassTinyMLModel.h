#ifndef MULTICLASS_TINYML_MODEL_H
#define MULTICLASS_TINYML_MODEL_H

#include <Arduino.h>
#include <stdint.h>
#include <math.h>

class MultiClassTinyMLModel {
public:
    static const int INPUT_DIM = 4;
    static const int HIDDEN_DIM = 8;
    static const int OUTPUT_DIM = 3;

    // Normalization parameters
    const float means[INPUT_DIM] = {5.84166667f, 3.04833333f, 3.77000000f, 1.20500000f};
    const float scales[INPUT_DIM] = {0.83741500f, 0.44665112f, 1.76113600f, 0.75947899f};

    // Layer 1 (Dense: 4 -> 8)
    const float W1[INPUT_DIM][HIDDEN_DIM] = {
        {-0.13709259f, 0.81185627f, 0.76185912f, 0.37523129f, -1.01308268f, -0.82943257f, -0.25934422f, 0.60530768f},
        {0.09286534f, -0.02318703f, -1.06104915f, 0.40035778f, 0.98487458f, -0.07112135f, -0.77008043f, -0.47416409f},
        {-0.23386671f, 0.47842351f, 0.45100264f, -0.02151575f, -0.33923913f, -0.88438439f, 0.12701033f, -0.48202512f},
        {-0.01902220f, 1.02838470f, 0.16991308f, 0.29657602f, -0.34096391f, -1.01216556f, 0.60225280f, -1.13525033f}
    };
    const float b1[HIDDEN_DIM] = {-0.65790177f, -0.24323615f, 1.19781528f, 0.62346385f, 0.48254521f, -0.22627913f, 0.15924464f, 1.03837944f};

    // Layer 2 (Dense: 8 -> 3)
    const float W2[HIDDEN_DIM][OUTPUT_DIM] = {
        {-0.52521598f, -0.04003742f, -0.71977209f},
        {0.31672023f, -1.17632662f, 0.66933919f},
        {-0.84871439f, 0.37105419f, 0.89370609f},
        {-0.44125609f, 0.70267184f, 0.28840595f},
        {1.23476672f, 0.04788902f, -0.53112121f},
        {1.07117419f, -1.12829475f, -0.77352387f},
        {-0.75678023f, -0.73215216f, 0.24110255f},
        {-0.06276946f, 1.50026274f, -1.50477237f}
    };
    const float b2[OUTPUT_DIM] = {-0.69648828f, 0.70072411f, -0.99644640f};

    int predict(const float *raw_input) {
        float normalized_x[INPUT_DIM];
        float hidden[HIDDEN_DIM];
        float output[OUTPUT_DIM];

        // 1. On-device Standard Scaling: (x - mean) / scale
        for (int i = 0; i < INPUT_DIM; i++) {
            normalized_x[i] = (raw_input[i] - means[i]) / scales[i];
        }

        // 2. Layer 1: Dense + ReLU
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float sum = b1[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += normalized_x[i] * W1[i][j];
            }
            hidden[j] = (sum > 0.0f) ? sum : 0.0f; // ReLU
        }

        // 3. Layer 2: Output Logits (8 -> 3)
        for (int k = 0; k < OUTPUT_DIM; k++) {
            float sum = b2[k];
            for (int j = 0; j < HIDDEN_DIM; j++) {
                sum += hidden[j] * W2[j][k];
            }
            output[k] = sum;
        }

        // 4. ArgMax over Logits to determine predicted class
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
};

static MultiClassTinyMLModel nnClassifier;

#endif
