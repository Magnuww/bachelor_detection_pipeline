pathToListOfImages = "C:\\Users\\vegar\\semester_6\\bachelor_project\\PythonSVM\\datasets\\Data_Bonafide\\bonafidePath.txt"
if __name__ == "__main__":
    with open(pathToListOfImages, 'r') as file:
        for line in file:
            lineWOStrip = line.strip()
            if lineWOStrip.endswith(".jpg"):
                newName = lineWOStrip.replace(".jpg", ".list")
                with open(newName, 'w') as file:
                    file.write(line)