import tensorflow as tf
import cv2
import numpy as np

import torch

from keypoints import HandKeypointDetector
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Încarcă modelul SavedModel
model_path = './'  # Directory containing saved_model.pb
model = tf.saved_model.load(model_path)

model_keypoints = HandKeypointDetector().to(device_torch)
model_keypoints.load_state_dict(torch.load("keypoints_checkpoints/keypoints_epoch62.pth", map_location=device_torch, weights_only=True))

# Get the signature for inference
infer = model.signatures['serving_default']

conf_treshold = 0.7

def predict_keypoints(image):
    # Preprocesare pentru modelul de keypoint detection
    image = cv2.resize(image, (256, 256))  # Resize to model's expected input size

    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device_torch) / 255.0
    with torch.no_grad():
        keypoints = model_keypoints(image_tensor)
    return keypoints.squeeze(0).cpu()

# Use eager execution instead of sessions
with tf.device('/CPU:0'):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocesare: convert to RGB and add batch dimension
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Keep as uint8 (quantized models expect 0-255 range)
        image_np_expanded = image_np_expanded.astype(np.uint8)
        
        # Run inference
        try:
            results = infer(tf.constant(image_np_expanded))
            
            # Extract results (adjust keys based on your model's output names)
            boxes = results['detection_boxes'].numpy()
            scores = results['detection_scores'].numpy()
            classes = results['detection_classes'].numpy()
            
            # Extrage prima mână detectată (cea mai sigură)
            if scores[0][0] > conf_treshold:
                ymin, xmin, ymax, xmax = boxes[0][0]
                
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixel coordinates
                left = int(xmin * w)
                top = int(ymin * h)
                right = int(xmax * w)
                bottom = int(ymax * h)

                cutout_padding = 20
                crop_left = max(0, left - cutout_padding)
                crop_top = max(0, top - cutout_padding)
                crop_right = min(w, right + cutout_padding)
                crop_bottom = min(h, bottom + cutout_padding)
                cutout = frame[crop_top:crop_bottom, crop_left:crop_right]
                keypoints = predict_keypoints(cutout)
                
                num_kp, Hm, Wm = keypoints.shape
                for i in range(num_kp):
                    hm = keypoints[i].detach().numpy()
                    idx = hm.argmax()
                    ym = idx // Wm
                    xm = idx % Wm
                    x_norm = xm / Wm
                    y_norm = ym / Hm
                    
                    # Convert to frame coordinates
                    kp_x = int(crop_left + x_norm * (crop_right - crop_left))
                    kp_y = int(crop_top + y_norm * (crop_bottom - crop_top))
                    
                    # Draw keypoint on frame
                    cv2.circle(frame, (kp_x, kp_y), 4, (0, 0, 255), -1)
                    cv2.circle(frame, (kp_x, kp_y), 5, (255, 0, 0), 1)
                
                # Draw bounding box
                cv2.rectangle(frame, (left, crop_top), (right, crop_bottom), (0, 255, 0), 2)
                
                # Draw confidence score
                confidence = scores[0][0]
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                



        except Exception as e:
            print(f"Inference error: {e}")

        # Try to display frame (may fail on headless systems)
        try:
            cv2.imshow('Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            print(f"Display not available: {e}")
            print("Running in headless mode - inference only")
            break
    
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # Display not available in headless mode