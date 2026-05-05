import os
import urllib.request

class Dataset:
    def __init__(self, testPath: str, trainPath: str, testLabels:str, trainLabels):
        self.TEST_PATH = testPath
        self.TEST_LABELS = testLabels
        self.TRAIN_PATH = trainPath
        self.TRAIN_LABELS = trainLabels


def createNonHumanLabel(labelPath: str):
    with open(labelPath, "w", encoding="utf-8") as f:
        f.write("1 0 0 0 0\n")

def populateNonHuman(dataset: Dataset):
    testImagesCount = len([1 for fileName in os.listdir(dataset.TEST_PATH)])
    trainImagesCount = len([1 for fileName in os.listdir(dataset.TRAIN_PATH)])

    testNonHumanCount = testImagesCount // 4
    trainNonHumanCount = trainImagesCount // 4

    totalNonHuman = testNonHumanCount + trainNonHumanCount
    
    nonhumanUrls = []

    with open("nonhuman_urls.txt", 'r') as f:
        nonhumanUrls = [f.readline().strip() for _ in range(totalNonHuman)]

    testImagesNHPath = os.path.join(dataset.TEST_PATH, "non_human")
    testLabelsNHPath = os.path.join(dataset.TEST_LABELS, "non_human")

    trainImagesNHPath = os.path.join(dataset.TRAIN_PATH, "non_human")
    trainLabelsNHPath = os.path.join(dataset.TRAIN_LABELS, "non_human")

    os.makedirs(testImagesNHPath, exist_ok=True)
    os.makedirs(testLabelsNHPath, exist_ok=True)
    os.makedirs(trainImagesNHPath, exist_ok=True)
    os.makedirs(trainLabelsNHPath, exist_ok=True)

    for i in range(testNonHumanCount):
        fileName = f"nonhuman_test_{i + 1}"
        imagePath = os.path.join(testImagesNHPath, f"nonhuman_test_{i + 1}.jpg")
        labelPath = os.path.join(testLabelsNHPath, f"{fileName}.txt")
        # urllib.request.urlretrieve(nonhumanUrls[i], imagePath)
        createNonHumanLabel(labelPath)

    for i in range(trainNonHumanCount):
        urlIndex = testNonHumanCount + i
        fileName = f"nonhuman_train_{i + 1}"
        imagePath = os.path.join(trainImagesNHPath, f"nonhuman_train_{i + 1}.jpg")
        labelPath = os.path.join(trainLabelsNHPath, f"{fileName}.txt")
        # urllib.request.urlretrieve(nonhumanUrls[urlIndex], imagePath)
        createNonHumanLabel(labelPath)

filteredDataset = Dataset(
    "dataset/filteredDataset/test/images_filtered",
    "dataset/filteredDataset/train/images_filtered",
    "dataset/filteredDataset/test/labels/YOLO_filtered",
    "dataset/filteredDataset/train/labels/YOLO_filtered"
)

populateNonHuman(filteredDataset)