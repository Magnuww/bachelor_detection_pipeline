import os
import sys
import morph_two_images as morph
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--folder",
                        type=str,
                        help="Path to the folder containing images")
    parser.add_argument("--morph",
                        type=str,
                        help="List of images containing morph relations")
    parser.add_argument("--out",
                        type=str,
                        help="output path")
    args = parser.parse_args()

    pathimages = args.folder
    pathmorph = args.morph


    #iterate over image paths in both folders

    #opens the morph file and passes the corresponding images to the morph two images.py script.
    with open(pathmorph, "r") as file:
        lines = file.readlines()
        for line in lines:
            img1, img2 = line.split()
            print(img1, img2)
            os.system(f"python3 morph_two_images.py --img1 {pathimages + img1} --img2 {pathimages + img2} --out {args.out}")

    """
    img1_folder = os.listdir(path1)
    img2_folder = os.listdir(path2)
    print(path1)
    print(img1_folder)
    for image1, image2 in zip(img1_folder, img2_folder):
        os.system(f"python3 morph_two_images.py --img1 {path1 + image1} --img2 {path2 + image2} --out {args.out}")
    """
