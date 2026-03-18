from detection.integral_image import compute_integral_image
from detection.haar_features import evaluate_window
from detection.sliding_window import sliding_window
import sys
import os

#starting parameters:
# size = 32
# step = 8
# threshold = -1
def detect_faces(image, step=16, threshold=0.25):
    integral = compute_integral_image(image)

    detections = []

    for scale in [32, 48, 64]:
        for (x, y, w, h) in sliding_window(image, scale, step):
            #feature = haar_feature_vertical(integral, x, y, w, h)
            score = evaluate_window(integral, x, y, w, h)

            if score > threshold:
                detections.append((x, y, w, h))

    return detections

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()