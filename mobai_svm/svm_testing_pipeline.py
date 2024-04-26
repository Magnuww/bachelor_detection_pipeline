from utils import readAllFeatureVectorsFromFolder
import numpy as np
import pickle
from libsvm_train_test import test_svm
from argparse import ArgumentParser


def testSVMModel(
    iSizeOfFeatureVector,
    strInputFeatureVectorFolder,
    strModelFilepath,
    strOutputFilePath,
):
    ##get all testing feature fectures
    nListOfAllFeatureVectors = readAllFeatureVectorsFromFolder(
        iSizeOfFeatureVector, strInputFeatureVectorFolder
    )
    ##load the pre-trained model
    loaded_model = pickle.load(open(strModelFilepath, "rb"))
    nListOfScores = loaded_model.predict_proba(nListOfAllFeatureVectors)
    arrayOfResults = np.asarray(nListOfScores)
    np.savetxt(strOutputFilePath, arrayOfResults, delimiter=",")


def addTrailingSlash(path):
    newPath = path
    if newPath[-1] != "/":
        newPath += "/"
    return newPath


def getInput():
    parser = ArgumentParser()
    parser.add_argument(
        "--bonaFideFeatures",
        type=str,
        help="Path to bonaFideFeatures",
        default="./Feature_Bonafide/",
    )
    parser.add_argument(
        "--morphedAttackFeatures",
        type=str,
        nargs="?",
        help="Path to attack features",
        default="./Feature_Morphed/",
    )

    parser.add_argument(
        "--modelOutput", type=str, help="output for model", default="./model_save/"
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

        test_svm(
            strInputBonafideFeaturesFolders,
            strInputAttacksFeaturesFolders,
            strSavingModelFilePath,
        )
