import glob, os, shutil

class Dataset:
    def __init__(self, testPath: str, trainPath: str, testLabels:str, trainLabels):
        self.TEST_PATH = testPath
        self.TEST_LABELS = testLabels
        self.TRAIN_PATH = trainPath
        self.TRAIN_LABELS = trainLabels

# Functie care renunta la imaginile care au mai mult de o mana in imagine
def filterHandDataset(dataset: Dataset):
    paths = [(dataset.TEST_LABELS, dataset.TEST_PATH), (dataset.TRAIN_LABELS, dataset.TRAIN_PATH)]

    for labelPath, imagePath in paths:
        filteredLabelsPath = f"{labelPath}_filtered"
        filteredImagesPath = f"{imagePath}_filtered"
        try:
            os.mkdir(filteredLabelsPath)
            os.mkdir(filteredImagesPath)
        except:
            pass
        
        for filename in glob.glob(os.path.join(labelPath, '*.txt')):
            with open(filename, 'r') as f:
                lineCount = sum(1 for _ in f)
                
                if lineCount == 1:
                    fileBasename = os.path.basename(filename)
                    imageFilename = fileBasename.split(".")[0] + ".jpg"
                    labelsDestPath = os.path.join(filteredLabelsPath, fileBasename)
                    shutil.copyfile(filename, labelsDestPath)

                    imagesDestPath = os.path.join(filteredImagesPath, imageFilename)
                    imageSourcePath = os.path.join(imagePath, imageFilename)
                    shutil.copyfile(imageSourcePath, imagesDestPath)
       
handDataset = Dataset(
    "dataset/test/images",
    "dataset/train/images",
    "dataset/test/labels/YOLO",
    "dataset/train/labels/YOLO"
)

filterHandDataset(handDataset)

