import os
from argparse import ArgumentParser

def split(path, file, split_size=100):
    rarr = []
    with open(path + file) as f:
        lines = f.readlines()
    for i in range(0, len(lines), split_size):
        if i + split_size > len(lines):
            with open(path + file + str(i), "w") as f:
                f.writelines(lines[i:])
            rarr.append(path + file + str(i))
        else:
            with open(path + file + str(i), "w") as f:
                f.writelines(lines[i : i + split_size])
            rarr.append(path + file + str(i))
    return rarr

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset", required=True)
    parser.add_argument("--output", type=str, help="output for images", required=True)
    parser.add_argument(
        "--script", type=str, help="Script for morphing images", required=True
    )
    parser.add_argument("--latent", type=str, help="Path to latent space", required=True)

    args = parser.parse_args()

    datasetPath = args.dataset
    outputPath = args.output
    latent = args.latent

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

    genders = [malesSource, femaleSource]

    for path in imagePaths:
        for gender in genders:
            csv = ""
            if gender == malesSource:
                csv = "1_male_source_morphing_list.csv"
            else:
                csv = "2_female_source_morphing_list.csv"
                
            morph_path = datasetPath + path + "/"
            input_path = datasetPath + path + "/" + gender + "/"

            csvsplit = split(morph_path, csv, 100)

            print("Morphing from: " + input_path)
            output = outputPath + path + "/" + gender
            if not os.path.exists(output):
                os.makedirs(output)
            for morph_list in csvsplit:
                print("Morphing: " + morph_list)
                os.system(
                    f"python3 {script} {morph_list} {input_path} {output} {latent} "
                )

    print("Finished morphing")
