from libsvm_train_test import train_svm, test_svm, tune_svm
import numpy as np

if __name__ == '__main__':
    
    # feature_sizes = np.array([(1024/1024, 1024/1), (1024/512, 1024/2), (1024/256, 1024/4),
    #                           (1024/128, 1024/8), (1024/64, 1024/16), (1024/32, 1024/32), 
    #                           (1024/16, 1024/64), (1024/8, 1024/128), (1024/4, 1024/256), 
    #                           (1024/2, 1024/512), (1024/1, 1024/1024)])
    
    feature_sizes = np.array([(int(1024/1024), int(1024/1)), (int(1024/512), int(1024/2)), (int(1024/256), int(1024/4)),
                          (int(1024/128), int(1024/8)), (int(1024/64), int(1024/16)), (int(1024/32), int(1024/32)),
                          (int(1024/16), int(1024/64)), (int(1024/8), int(1024/128)), (int(1024/4), int(1024/256)),
                          (int(1024/2), int(1024/512)), (int(1024/1), int(1024/1024))])


    # for i, value in enumerate(feature_sizes):
    #     variable = tuple(feature_sizes[i])
    #     break
    
    
    feature_type = "ArcFaceFeatures"
    # featureType = "CurricularFaceFeatures"
    # featureType = "ElasticFaceFeatures"
    # featureType = "MagFaceFeatures"
    for j, value in enumerate(feature_sizes):
        variable = tuple(feature_sizes[j])
        strInputBonafideFeaturesFolders = f"./datasets/allFaceFeatures/allFaceFeatures/{feature_type}/Feature_Bonafide/" 
        strInputAttacksFeaturesFolders = f"./datasets/allFaceFeatures/allFaceFeatures/{feature_type}/Feature_Morphed/"
        
        param_strs = [
            ['-s 0 -t 0 -c 10 -b 1 -q', True],
            # ['-s 0 -t 0 -b 1 -q', True],
        ]
        for i, param_i in enumerate(param_strs):
            print("Starting: " + param_i[0] + '_' + str(param_i[-1]))
            os_i = param_i[-1]
            str_i = param_i[0]
            strSavingModelFilePath = f'./PythonSVM/concatenate/model_{str_i}_os_{os_i}/size_comb_{j}/'

            # tune_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, "svm_tuning")
            train_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, 
                    strSavingModelFilePath, feature_shape = variable, param_str=str_i, oversampled=os_i)
            
            test_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, strSavingModelFilePath, feature_shape=variable)   
