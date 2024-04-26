import os
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser


def getAllSubFolders(strInputFolder):
    nListOfSubFolderNames = os.listdir(strInputFolder)
    return nListOfSubFolderNames


def getAllFilesInFolder(strInputFolder, strFileExtension):
    nListOfAllFiles = [
        f for f in os.listdir(strInputFolder) if f.endswith(strFileExtension)
    ]
    return nListOfAllFiles


def getAllFilesRegardlessExtension(strInputFolder):
    nListOfFiles = [
        f for f in listdir(strInputFolder) if isfile(join(strInputFolder, f))
    ]
    return nListOfFiles


def getAllFiles_start_with_string(strInputFolder, strFileExtension):
    nListOfAllFiles = [
        f for f in os.listdir(strInputFolder) if f.startswith(strFileExtension)
    ]
    return nListOfAllFiles


def saveListValuesIntoFile(strSavingFilePath, nListOfValues):
    with open(strSavingFilePath, "w") as fileHandle:
        for eachValue in nListOfValues:
            fileHandle.write("%s\n" % eachValue)


def readValuesFromFile(strInputFilePath, print_flag=True):
    ##np.loadtxt(fileName, dtype=float)
    featureVector = []
    with open(strInputFilePath) as file:
        if print_flag:
            print(strInputFilePath)
        for line in file:
            # print(line.rstrip())
            strEachValue = line.rstrip()
            featureVector.append(float(strEachValue))
    return featureVector


def readAllFeatureVectorsFromFolder(iSizeOfFeatureVector, strInputFolder):
    nListOfSubFolders = getAllSubFolders(strInputFolder)
    nListOfFeatureVectors = []
    for strEachSubFolder in nListOfSubFolders:
        if strEachSubFolder == ".DS_Store":
            continue
        strEachSubFolderFullPath = os.path.join(strInputFolder, strEachSubFolder)
        nListOfallFile_subFolder = getAllFilesRegardlessExtension(
            strEachSubFolderFullPath
        )
        for strEachFile in nListOfallFile_subFolder:
            if strEachFile == ".DS_Store":
                continue
            strEachFileFullPath = os.path.join(strEachSubFolderFullPath, strEachFile)
            nEachFeatureVector = readValuesFromFile(strEachFileFullPath)
            if len(nEachFeatureVector) == iSizeOfFeatureVector:
                nListOfFeatureVectors.append(nEachFeatureVector)
    return nListOfFeatureVectors


def addTrailingSlash(path):
    newPath = path
    if newPath[-1] != "/":
        newPath += "/"
    return newPath


def getUserArgs():
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
