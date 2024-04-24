import os
from utils import readAllFeatureVectorsFromFolder
import numpy as np
from sklearn import svm
import pickle
from libsvm_train_test import test_svm


def testSVMModel(iSizeOfFeatureVector, strInputFeatureVectorFolder, strModelFilepath, strOutputFilePath):
    ##get all testing feature fectures
    nListOfAllFeatureVectors = readAllFeatureVectorsFromFolder(iSizeOfFeatureVector, strInputFeatureVectorFolder)
    ##load the pre-trained model
    loaded_model = pickle.load(open(strModelFilepath, 'rb'))
    nListOfScores = loaded_model.predict_proba(nListOfAllFeatureVectors)
    arrayOfResults = np.asarray(nListOfScores)
    np.savetxt(strOutputFilePath, arrayOfResults, delimiter=",")
    i = 0


if __name__ == '__main__':
    print('PyCharm')
    iSizeOfFeatureVector = 25088 
    strInputBonafideFeaturesFolders = "../Feature_Bonafide/"
    strInputMorphFeaturesFolders = "../Feature_Morphed/"
    strSavingModelFilePath = "./model_save/"
    strOutputFilePath = "./model_save/predictions.csv"
    test_svm(iSizeOfFeatureVector, strInputBonafideFeaturesFolders, strInputMorphFeaturesFolders, strSavingModelFilePath, strOutputFilePath)



