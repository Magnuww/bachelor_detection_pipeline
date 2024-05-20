import os
import shutil as sh
from argparse import ArgumentParser

argparse = ArgumentParser()
argparse.add_argument("--bonafied", help="bonafied folder")
argparse.add_argument("--morph", help="output folder")

args = argparse.parse_args()

# traverse the file system with os.walk
for root, dirs, files in os.walk(args.morph):
    for file in files:
        # check if the file is in the list of files
        prefix = file.split(".")
        imgs = prefix[0].split("_")
        img1 = imgs[1]
        img2 = imgs[2]

        lenpassed = len(args.bonafied.strip("/").split("/"))

        path = root.split("/")

        bonafiedpath1 = (
            args.bonafied + "/" + "/".join(path[lenpassed:]) + "/" + f"probe_{img1}.txt"
        )
        bonafiedpath2 = (
            args.bonafied + "/" + "/".join(path[lenpassed:]) + "/" + f"probe_{img2}.txt"
        )
        newpath = root + "/" + f"probe_{img1}.txt"
        newpath2 = root + "/" + f"probe_{img2}.txt"
        path2 = os.path.exists(bonafiedpath2)
        path1 = os.path.exists(bonafiedpath1)
        if path1:
            sh.copy(bonafiedpath1, newpath)
        if path2:
            sh.copy(bonafiedpath2, newpath2)
