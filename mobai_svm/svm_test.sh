bonaFideFeatures="../datasets/unibo_w_mordiff_test/runnable/Feature_Bonafide/"
morphedAttackFeatures="../datasets/unibo_w_mordiff_test/runnable/Feature_Morphed/"

modelOutput="../models/unibo_morph_w_mordiff_test/PythonSVM"

python3 svm_testing_pipeline.py --bonaFideFeatures $bonaFideFeatures \
	--morphedAttackFeatures $morphedAttackFeatures \
	--modelOutput $modelOutput
