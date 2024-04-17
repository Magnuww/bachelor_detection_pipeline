from libsvm_train_test import train_svm, test_svm, tune_svm
import argparse as ap
import dual_test

parse = ap.ArgumentParser()

parse.add_argument("--Mfeatures", type=str, help="Path to bonafide features folder.")
parse.add_argument("--Ffeatures", type=str, help="Path to morphed features folder.")
parse.add_argument("--Mname", type=str, help="Path to save mobaimodels.")
parse.add_argument("--Fname", type=str, help="Path to save .")
parse.add_argument("--Trainmodels", type=bool, help="Skip training mobai models.")

args = parse.parse_args()


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
        strSavingModelFilePath = f'./PythonSVM/concatenate/model_{args.Mname}_os_{os_i}/'
        strSavingModelFilePath2 = f'./PythonSVM/concatenate/model_{args.Fname}_os_{os_i}/'

        print("what" + str(args.Trainmodels))
        if args.Trainmodels:
            # tune_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, "svm_tuning")
            train_svm(args.Mfeatures + "/" + strInputBonafideFeaturesFolders, args.Mfeatures + "/" + strInputAttacksFeaturesFolders, 
                    strSavingModelFilePath, param_str=str_i, oversampled=os_i, feat_shapes=(49,512))
            print("Finished training movai svm")
            train_svm(args.Ffeatures + "/" + strInputBonafideFeaturesFolders, args.Ffeatures + "/" + strInputAttacksFeaturesFolders, 
                    strSavingModelFilePath2, param_str=str_i, oversampled=os_i, feat_shapes=(32,32))
            print("Done Training")
        print("testing models")
        dual_test.dual_test_svm(args.Mfeatures + "/" + strInputBonafideFeaturesFolders, args.Mfeatures + "/" + strInputAttacksFeaturesFolders, strSavingModelFilePath, (49,512),
             args.Ffeatures + "/" + strInputBonafideFeaturesFolders, args.Ffeatures + "/" + strInputAttacksFeaturesFolders, strSavingModelFilePath2,(32,32))
