from libsvm_train_test import train_svm, test_svm, tune_svm
import argparse as ap
from nntraintest import test_nn, train_nn
from rftraintest import train_test_rf
parse = ap.ArgumentParser()

parse.add_argument("--Ffeatures", type=str, help="Path to Mobai features folder.")
parse.add_argument("--Fname", type=str, help="Path to save mobaimodels.")
parse.add_argument("--Train_model", type=bool, help="Train models.")
parse.add_argument("--Tune_ratio", type=bool, help="Tune ratio")
parse.add_argument("--plot_name", type=str, help="Name of the plot.", default="")

args = parse.parse_args()


if __name__ == '__main__':
    strInputBonafideFeaturesFolders = "./Feature_Bonafide/" 
    strInputAttacksFeaturesFolders = "./Feature_Morphed/"

    param_strs = [
        ['-s 0 -t 0 -c 10 -b 1 -q', True],
        # ['-s 0 -t 0 -b 1 -q', True],
    ]

    # shape1 = (49,512)
    shape1=(1,25088)
    # shape1=(1,1024)
    for i, param_i in enumerate(param_strs):
        print("Starting: " + param_i[0] + '_' + str(param_i[-1]))
        os_i = param_i[-1]
        str_i = param_i[0]

        strSavingModelFilePath = f'./magnusnn/concatenate/model_{args.Fname}_os_{os_i}/'

        print("what" + str(args.Train_model))
        if args.Train_model:
            # tune_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, "svm_tuning")
            StrInputBonafideFeaturesFolders = args.Ffeatures + "/" + strInputBonafideFeaturesFolders
            StrInputAttacksFeaturesFolders = args.Ffeatures + "/" + strInputAttacksFeaturesFolders

            train_nn(
                strInputBonafideFeaturesFolders=StrInputBonafideFeaturesFolders,
                strInputAttacksFeaturesFolders= StrInputAttacksFeaturesFolders,
                strSavingModelFilePath=strSavingModelFilePath,
                param_str=str_i,
                oversampled=os_i, 
                feat_shapes=shape1,
                )
            # train_svm(args.Mfeatures + "/" + strInputBonafideFeaturesFolders, args.Mfeatures + "/" + strInputAttacksFeaturesFolders, 
            #         strSavingModelFilePath, param_str=str_i, oversampled=os_i, feat_shapes=(49,512))
            # print("Finished training movai svm")
            # train_svm(args.Ffeatures + "/" + strInputBonafideFeaturesFolders, args.Ffeatures + "/" + strInputAttacksFeaturesFolders, 
            #         strSavingModelFilePath2, param_str=str_i, oversampled=os_i, feat_shapes=(1,1024))
            # print("Done Training")
        # print("testing models")
        # tune_ratio = True if args.Tune_ratio else False
        # if tune_ratio:
        #     dual_tune_ratio(args.Mfeatures + "/" + strInputBonafideFeaturesFolders, args.Mfeatures + "/" + strInputAttacksFeaturesFolders, strSavingModelFilePath, shape1,
        #         args.Ffeatures + "/" + strInputBonafideFeaturesFolders, args.Ffeatures + "/" + strInputAttacksFeaturesFolders, strSavingModelFilePath2,shape2, plotname=args.plot_name)
        # else:
        test_nn(args.Ffeatures + "/" + strInputBonafideFeaturesFolders, args.Ffeatures + "/" + strInputAttacksFeaturesFolders, strSavingModelFilePath,shape1,plotname=args.plot_name)
        # print("HEI")
