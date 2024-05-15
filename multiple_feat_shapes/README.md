- Important note regarding SVMs: 
	- The master branch use SVMs that take concatenated features for a probe and ref as input. Currently best model use oversampling and C=20 regularisation: model_-s 0 -t 0 -c 10 -b 1 -q_os_True
	- To use unsigned diff (i.e. feats_probe - feats_ref), switch to the "diff" branch. Current model use oversampling and C=1 (default) regularisation: model_-s 0 -t 0 -b 1 -q_os_True  

# Python scripts to train/tune/test SVMs using libsvm

## Why libsvm? --> Models can be reused in cpp. 
- The models that are used have the setup ```model_-s 0 -t 0 -c 20 -b 1 -q_os_True```. 
	- i.e. Linear with probability estimates, oversampling and c=20 regularisation 

## Some notes on tuning: 
- Class imbalance (dealing with about twice as many morphed as bonafide samples)
	- Nothing --> Baseline
	- Weighting of loss function --> Improved slightly
	- Undersampling of majority class (morphed) --> Improved
	- Oversampling of minory class (bonafide) --> Worked the best 
- Regularisation: 
	- Nothing --> Baseline
	- C=0.1 --> Worse
	- C=10 --> Much better
	- C=20 --> Slightly better than C=10
- Kernel (only limited testing)
	- Linear --> Baseline and the one used
	- Polynomial 3rd degree --> Slightly worse
	- RBF --> Worse
- Future work: 
	- larger and more balanced dataset
	- Further testing of different non-lineear kernels. 
	- PCA or other dimensionality reduction to reduce the number of input features. 

## File structure: 
- model_XXX folders: 
	- These folder contain the different saved SVM models and test results. 
	- Each folder contain 49 different models (one for each latent FRS feature). 
	- To use the models in the cpp code, copy the 49 model files in a specific folder to the root/morph/config/svm folder. 
- svm_tuning folder: 
	- Stores the results from tuning in the libsvm_train_test.py file. 
libsvm_train_test.py: 
	- Function with the tuning, training and test functionality. 
svm_training_pipeline.py: 
	- Function that run tuning, trainin or testing from the libsvm_train_test.py file. 
svm_testing_pipeline.py: 
	- Not currently in use as all the functionality is contained in svm_training_pipeline.py.
- data_loader.py: 
	- python code to define a data loader to construct dataset with probe and reference image pairs given from the data in root/Feature_Bonafide and root/Feature_Morphed folders. 
	- The data_loader is used to construct and load the different datasets. 
- utils.py: 
	- Contains some core functionality. 
- checkmissing.py: 
	- Old file with a structure to check the number of files or missing files. Not currently in use. 


## Future Work: 
- Currently the number of input features to the SVMs is fairly large compared to the dataset size, i.e. roughly 7k input samples each with 512*2=1024 input features. For future work it would be interesting to investigate methods to reduce the dimensionality through e.g: 
	- Feature reduction of latent FRS features through PCA.
	- Train a enc-dec or VAE on the latent features. Encoder to reduce dimensionality and decoder to reconstruct original input from lower dimensionality. Then the lower-dim features might be used as inputs instead of the full 512 or 1024 feature vectors. 
- Alternatively, it would be good to increase the size of the dataset. 
- Furthermore, the class imbalance is somewhat an issue. Currently, we found oversampling of the minority class to work well to artificially balance the dataset, but it might be better to get more bonafide samples. 