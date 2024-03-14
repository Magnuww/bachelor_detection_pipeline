from libsvm_train_test import train_svm, test_svm, tune_svm

if __name__ == '__main__':
    strInputBonafideFeaturesFolders = "./Feature_Bonafide/" 
    strInputAttacksFeaturesFolders = "./Feature_Morphed/"

    param_strs = [
        ['-s 0 -t 0 -c 10 -b 1 -q', True],
        # ['-s 0 -t 0 -b 1 -q', True],
    ]
    for i, param_i in enumerate(param_strs):
        print("Starting: " + param_i[0] + '_' + str(param_i[-1]))
        os_i = param_i[-1]
        str_i = param_i[0]
        strSavingModelFilePath = f'./PythonSVM/concatenate/model_{str_i}_os_{os_i}/'

        # tune_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, "svm_tuning")
        train_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, 
                  strSavingModelFilePath, param_str=str_i, oversampled=os_i)
        test_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath)   
