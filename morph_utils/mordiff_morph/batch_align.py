import os
from argparse import ArgumentParser

if __name__ == "__main__":
    # NOTE: required arguments are usually bad form, but we dont care in this case
    # https://docs.python.org/3/library/argparse.html#required
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset", required=True)
    parser.add_argument("--output", type=str, help="output for images", required=True)
    parser.add_argument(
        "--script", type=str, help="Script for aligning images", required=True
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
    malesSourceName = "1_male_source"
    femaleSourceName = "2_female_source"
    imagePaths = []
    subLevel1 = os.listdir(datasetPath)

    for entry in subLevel1:
        subLevel2 = os.listdir(datasetPath + entry)
        for entry2 in subLevel2:
            subLevel3 = os.listdir(datasetPath + entry + "/" + entry2)
            imagePaths.append(entry + "/" + entry2 + "/" + malesSourceName)
            imagePaths.append(entry + "/" + entry2 + "/" + femaleSourceName)

    for path in imagePaths:
        print("Aligning from: " + path)
        output = outputPath + path
        # creating directories
        if not os.path.exists(output):
            os.makedirs(output)

        inputPath = datasetPath + path
        os.system(
            f"python3 {script} --input_imgs_path {inputPath} --output_imgs_path {output}"
        )

    print("Finished aligning")
