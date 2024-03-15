from libsvm_train_test import train_svm, test_svm, tune_svm
from argparse import ArgumentParser


def addTrailingSlash(path):
    newPath = path
    if newPath[-1] != "/":
        newPath += "/"
    return newPath


def getInput():
    parser = ArgumentParser()
    parser.add_argument(
        "--bonaFideFeatures", type=str, help="Path to bonaFideFeatures", required=True
    )
    parser.add_argument(
        "--morphedAttackFeatures",
        type=str,
        help="Path to attack features",
        required=True,
    )

    parser.add_argument(
        "--modelOutput",
        type=str,
        help="output for model",
        required=True,
    )

    args = parser.parse_args()

    strInputBonafideFeaturesFolders = addTrailingSlash(args.bonaFideFeatures)
    strInputAttacksFeaturesFolders = addTrailingSlash(args.morphedAttackFeatures)
    modelOutput = addTrailingSlash(args.modelOutput)
    return strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, modelOutput


if __name__ == "__main__":
    strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, modelOutput = (
        getInput()
    )

    param_strs = [
        ["-s 0 -t 0 -c 10 -b 1 -q", True],
        # ['-s 0 -t 0 -b 1 -q', True],
    ]
    for i, param_i in enumerate(param_strs):
        print("Starting: " + param_i[0] + "_" + str(param_i[-1]))
        os_i = param_i[-1]
        str_i = param_i[0]
        # strSavingModelFilePath = f"./PythonSVM/concatenate/model_{str_i}_os_{os_i}/"
        strSavingModelFilePath = f"{modelOutput}concatenate/model_{str_i}_os_{os_i}/"

        # tune_svm(strInputBonafideFeaturesFolders, strInputAttacksFeaturesFolders, "svm_tuning")
        train_svm(
            strInputBonafideFeaturesFolders,
            strInputAttacksFeaturesFolders,
            strSavingModelFilePath,
            param_str=str_i,
            oversampled=os_i,
        )
        test_svm(
            strInputBonafideFeaturesFolders,
            strInputAttacksFeaturesFolders,
            strSavingModelFilePath,
        )
