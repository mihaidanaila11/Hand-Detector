
from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Încarcă fișierul de adnotări
TRAIN_LABELS = Path('dataset/oxford_hands/training_dataset/training_data/annotations')
CONVERTED_LABELS = TRAIN_LABELS / "converted"


CONVERTED_LABELS.mkdir(exist_ok=True)


for labelPath in TRAIN_LABELS.iterdir():
    if labelPath.suffix.lower() != '.mat':
        continue

    print(labelPath)

    labelData = loadmat(labelPath)
    boxes = np.array(labelData["boxes"]).squeeze()

    if boxes.ndim == 0:
        boxes = [boxes.item()]
    
    with open(CONVERTED_LABELS / f"{labelPath.stem}.txt", "w") as f:
        for box in boxes:
            pointsTuple = box[0][0].item()[:4]
            
            points = np.concatenate(pointsTuple)

            y_coords = points[:, 0]
            x_coords = points[:, 1]

            xmin = np.min(x_coords)
            ymin = np.min(y_coords)
            xmax = np.max(x_coords)
            ymax = np.max(y_coords)
            
            line = f"1 {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n"
            f.write(line)
