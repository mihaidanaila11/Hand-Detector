import cv2
import torch
from torchvision import transforms
from PIL import Image
from model2 import HandDetectionNetwork
from itertools import product
from math import sqrt

class AnchorGenerator:
    def __init__(self, image_size=300):
        self.image_size = image_size
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2, 3]] * len(self.feature_maps)

    def generate(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                s_k = self.min_sizes[k] / self.image_size
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))

                anchors.append([cx, cy, s_k, s_k])

                anchors.append([cx, cy, s_k_prime, s_k_prime])

                for ar in self.aspect_ratios[k]:
                    anchors.append([cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]) 
                    anchors.append([cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]) 

        output = torch.tensor(anchors).view(-1, 4)
        
        output.clamp_(max=1, min=0) 
        
        return output
  

image_size = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cxcy_to_xy(cxcy):
    #transformă [cx, cy, w, h] în colțuri [xmin, ymin, xmax, ymax]
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def decode_boxes(loc_preds, priors):
    variance = [0.1, 0.2]
    
    boxes_cxcy = torch.cat((
        priors[:, :2] + loc_preds[:, :2] * variance[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc_preds[:, 2:] * variance[1])
    ), 1)
    return boxes_cxcy


model = HandDetectionNetwork(num_anchors=6).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.eval() 


anchor_gen = AnchorGenerator(image_size=image_size)
default_boxes = anchor_gen.generate().to(device) 


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False

print("Apasă ESC pentru a ieși.")

while rval:
    H, W, _ = frame.shape

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor) 
        
        preds_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1) 
                               for o in out["cls"]], dim=1) 
        preds_loc = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1, 4) 
                               for o in out["loc"]], dim=1) 
        probs = torch.sigmoid(preds_cls.squeeze(0)) 

        best_prob, best_idx = probs.max(dim=0)
        
        if best_prob.item() > 0.5:
            
            best_loc = preds_loc[0, best_idx, :].unsqueeze(0)
            best_prior = default_boxes[best_idx, :].unsqueeze(0)
            
            decoded_box_cxcy = decode_boxes(best_loc, best_prior)
            decoded_box_xy = cxcy_to_xy(decoded_box_cxcy).squeeze(0)

            xmin = int(decoded_box_xy[0].item() * W)
            ymin = int(decoded_box_xy[1].item() * H)
            xmax = int(decoded_box_xy[2].item() * W)
            ymax = int(decoded_box_xy[3].item() * H)

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Mana: {best_prob.item()*100:.1f}%", (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("preview", frame)
    
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: 
        break

cv2.destroyWindow("preview")
vc.release()