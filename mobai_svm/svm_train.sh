bonaFideFeatures="../datasets/unibo/runnable/Feature_Bonafide/"
# morphedAttackFeatures="../datasets/mordiff/full_morphed_features/Feature_Morphed/"
morphedAttackFeatures="../datasets/mordiff/runnable3/Feature_Morphed/"

modelOutput="../models/mordiff_morph_original_w_unibo_test/

python3 svm_training_pipeline.py --bonaFideFeatures $bonaFideFeatures \
	--morphedAttackFeatures $morphedAttackFeatures \
	--modelOutput $modelOutput
