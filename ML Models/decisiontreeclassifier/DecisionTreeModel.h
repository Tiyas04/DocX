#ifndef UUID2636262774432
#define UUID2636262774432

/**
  * RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_name=RandomForestClassifier, class_weight=None, criterion=gini, estimator=DecisionTreeClassifier(), estimator_params=('criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'random_state', 'ccp_alpha', 'monotonic_cst'), max_depth=4, max_features=sqrt, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, monotonic_cst=None, n_estimators=1, n_jobs=None, num_outputs=2, oob_score=False, package_name=everywhereml.sklearn.ensemble, random_state=42, template_folder=everywhereml/sklearn/ensemble, verbose=0, warm_start=False)
 */
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