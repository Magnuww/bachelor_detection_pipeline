import os
import sys
import numpy as np
from argparse import ArgumentParser
import re
#Different regex for different databases naming conventions
tuf_rename = re.compile(r'M_(\d+)_(\d+).*')
age_rename = re.compile(r'M_(\d+)_.*[M|F].*_(\d+)_.*_W0.*')
feret_rename = re.compile(r'M_(\d+)_1_(\d+).*')
frgc_rename = re.compile(r'M_(\d+)d.*_(\d+)d.*')

def format_name(root : str, file: str) -> str :
    path = root.split("/")
    argspassed = len(args.out.split("/"))
    database = path[argspassed]



    match = "" 
    # print(database)
    match database:
        case "FERET":
            match = feret_rename.match(file)
        case "FRGC":
            match = frgc_rename.match(file)
        case "TUF":
            match = tuf_rename.match(file)
        case "AGE":
            match = age_rename.match(file)
    if match == "":
        raise ValueError("No match found")
    # print(file)
    # print(match.groups())
    new_name = 'ref_{}_{}.txt'.format(*match.groups())

    return new_name


def traverse(path, out : str, rename=False):
    for root, dirs, files in os.walk(path):
        #print(root, dirs, files)
        for file in files:
            if file.endswith(".npy"):
                #print(file, root, dirs)
                path = root.split("/")
                #print("path")
                #print(path)
                #print(out)
                outpath = out + "/" + "/".join(path[-5:-2]) + "/"
                #print(outpath)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                data = np.load(root + "/" + file)

                name = file[:-4] + ".txt"
                
                #print(name)

                if rename:
                    name = format_name(root, file)
                    # print(name)
                    # break
                #print("outpath + name")
                print(outpath + name)

                with open(outpath + name, "w") as f:
                    for i in data:
                        pass
                        f.write(str(i) + "\n")

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--path",
                        type=str,
                        help="Path to the folder containing images")
    parser.add_argument("--out",
                        type=str,
                        help="output path")
    parser.add_argument("--rename",
                        type=bool,
                        help="rename files")
    args = parser.parse_args()

    if args.rename:
        traverse(args.path, args.out, True)
    else:
        traverse(args.path, args.out)




        #os.system("python3 feature_extraction/extract_emb.py ")
    #iterate over image paths in both folders

    #opens the morph file and passes the corresponding images to the morph two images.py script.
    #    with open(pathmorph, "r") as file:
    #        lines = file.readlines()
    #        for line in lines:
    #            img1, img2 = line.split()
    #            print(img1, img2)
    #            os.system(f"python3 morph_two_images.py --img1 {pathimages + img1} --img2 {pathimages + img2} --out {args.out}")

    """
    img1_folder = os.listdir(path1)
    img2_folder = os.listdir(path2)
    print(path1)
    print(img1_folder)
    for image1, image2 in zip(img1_folder, img2_folder):
        os.system(f"python3 morph_two_images.py --img1 {path1 + image1} --img2 {path2 + image2} --out {args.out}")
    """
