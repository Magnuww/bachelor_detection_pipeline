import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset", required=True)
    parser.add_argument("--output", type=str, help="output for images", required=True)
    parser.add_argument(
        "--script", type=str, help="Script for morphing images", required=True
    )

    args = parser.parse_args()

    datasetPath = args.dataset
    outputPath = args.output

    datasetPath = os.path.abspath(args.dataset)
    outputPath = os.path.abspath(args.output)

    if datasetPath[-1] != "/":
        datasetPath += "/"

    if outputPath[-1] != "/":
        outputPath += "/"

    script = args.script

    # file structure for dataset
    imagePaths = []
    subLevel1 = os.listdir(datasetPath)

    # TODO: rename sublevels
    for entry in subLevel1:
        subLevel2 = os.listdir(datasetPath + entry)
        for entry2 in subLevel2:
            imagePaths.append(entry + "/" + entry2)

    malesSource = "1_male_source"
    femaleSource = "2_female_source"
    morphingList = "_morphing_list.txt"

    genders = [malesSource, femaleSource]

    for path in imagePaths:
        for gender in genders:
            morphList = datasetPath + path + "/" + gender + morphingList
            inputPath = datasetPath + path + "/" + gender + "/"
            print("Morphing from: " + inputPath)
            output = outputPath + path + "/" + gender
            if not os.path.exists(output):
                os.makedirs(output)

            with open(morphList, "r") as file:
                lines = file.readlines()
                for line in lines:
                    img1, img2 = line.split()
                    # checking if image already exists
                    imgName1 = img1.split(".")[0]
                    imgName2 = img2.split(".")[0]
                    fileName = "_" + imgName1 + "_" + imgName2 + ".png"

                    if os.path.exists(output + "/" + fileName):
                        print("The file " + fileName + " already exist")
                        continue
                    print("Morphing: " + img1 + ", " + img2)
                    os.system(
                        f"python3 {script} --img1 {inputPath + img1} --img2 {inputPath + img2} --out {output}"
                    )

    print("Finished morphing")
