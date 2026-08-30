#ifndef UUID1963415934624
#define UUID1963415934624
#include <stdint.h>
#include <Arduino.h>
class RandomForestClassifier {
    public:

        /**
         * Predict class from features
         */
        int predict(float *x) {
            int predictedValue = 0;
            size_t startedAt = micros();

            
                    
            float votes[2] = { 0 };
            uint8_t classIdx = 0;
            float classScore = 0;

            
                tree0(x, &classIdx, &classScore);
                votes[classIdx] += classScore;
            

            uint8_t maxClassIdx = 0;
            float maxVote = votes[0];

            for (uint8_t i = 1; i < 2; i++) {
                if (votes[i] > maxVote) {
                    maxClassIdx = i;
                    maxVote = votes[i];
                }
            }

            predictedValue = maxClassIdx;

                    

            latency = micros() - startedAt;

            return (lastPrediction = predictedValue);
        }

        
            
            /**
             * Get latency in micros
             */
            uint32_t latencyInMicros() {
                return latency;
            }

            /**
             * Get latency in millis
             */
            uint16_t latencyInMillis() {
                return latency / 1000;
            }
            

    protected:
        float latency = 0;
        int lastPrediction = 0;

        
            
        
            
                /**
                 * Random forest's tree #0
                 */
                void tree0(float *x, uint8_t *classIdx, float *classScore) {
                    
                        if (x[1] < 0.09276334568858147) {
                            
                        if (x[0] < -0.3390306681394577) {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }
                        else {
                            
                        if (x[1] < -0.012043268419802189) {
                            
                        *classIdx = 1;
                        *classScore = 0.46;
                        return;

                        }
                        else {
                            
                        if (x[1] < -0.002830510726198554) {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }
                        else {
                            
                        *classIdx = 1;
                        *classScore = 0.46;
                        return;

                        }

                        }

                        }

                        }
                        else {
                            
                        if (x[1] < 0.7673953473567963) {
                            
                        if (x[0] < -0.512126550078392) {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }
                        else {
                            
                        if (x[0] < 0.5841663032770157) {
                            
                        *classIdx = 1;
                        *classScore = 0.46;
                        return;

                        }
                        else {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }

                        }

                        }
                        else {
                            
                        if (x[1] < 0.8664935529232025) {
                            
                        if (x[1] < 0.8519493341445923) {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }
                        else {
                            
                        *classIdx = 1;
                        *classScore = 0.46;
                        return;

                        }

                        }
                        else {
                            
                        *classIdx = 0;
                        *classScore = 0.54;
                        return;

                        }

                        }

                        }

                }
            
        

            
};



static RandomForestClassifier treeClassifier;


#endif