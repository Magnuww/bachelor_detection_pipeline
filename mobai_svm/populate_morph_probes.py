import os
import shutil as sh
from argparse import ArgumentParser

argparse = ArgumentParser()
argparse.add_argument('--bonafied', help='bonafied folder')
argparse.add_argument('--morph', help='output folder')

args = argparse.parse_args()

#traverse the file system with os.walk
for root, dirs, files in os.walk(args.morph):
    for file in files:
        #check if the file is in the list of files
        #root print(file)
        print(file)
        prefix = file.split(".")
        imgs  = prefix[0].split("_")
        # print(imgs)
        img1 = imgs[1]
        img2 = imgs[2]

        lenpassed = len(args.bonafied.strip("/").split("/"))
        print(lenpassed)

        path = root.split("/")
        print(path)
        print(path[lenpassed:])

        bonafiedpath1 = args.bonafied + "/" + "/".join(path[lenpassed:]) + "/" + f"probe_{img1}.txt"
        bonafiedpath2 = args.bonafied + "/" + "/".join(path[lenpassed:]) + "/" + f"probe_{img2}.txt"
        newpath = root + "/" + f"probe_{img1}.txt"
        newpath2 = root + "/" + f"probe_{img2}.txt"
        print(os.getcwd())
        print(newpath)
        print(newpath2)
        path2 = os.path.exists(bonafiedpath2)
        path1 = os.path.exists(bonafiedpath1)
        print(bonafiedpath1, path1)
        print(bonafiedpath2, path2)
        if path1:
            # print(root)
            sh.copy(bonafiedpath1, newpath)
        if path2:
            sh.copy(bonafiedpath2, newpath2)

