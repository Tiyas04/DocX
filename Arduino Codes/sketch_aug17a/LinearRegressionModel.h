#ifndef LINEAR_REGRESSION_MODEL_H
#define LINEAR_REGRESSION_MODEL_H

class LinearRegressionModel {
public:
    const float slope = 2.48840332f;
    const float intercept = 3.01288862f;

    float predict(float *x) {
        return (slope * x[0]) + intercept;
    }
};

LinearRegressionModel linReg;

#endif
