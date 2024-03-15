bonaFideFeatures="../datasets/unibo/runnable/Feature_Bonafide/"
morphedAttackFeatures="../datasets/mordiff/runnable/"

modelOutput="../models/mordiff_morph_2/"

python3 svm_training_pipeline.py --bonaFideFeatures $bonaFideFeatures \
	--morphedAttackFeatures $morphedAttackFeatures \
	--modelOutput $modelOutput
