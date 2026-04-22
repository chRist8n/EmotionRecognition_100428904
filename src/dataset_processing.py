import os
import cv2
import numpy as np

from landmarking.landmark_detector import detect_landmarks
from landmarking.feature_extraction import extract_features

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)

#############################################################

"""
NOTE:   Label mappings are changed from the originals since 
        i feel it makes better sense this way

Original mappings:   -->    New mappings:
1 - Surprise                0 - Neutral
2 - Fear                    1 - Happy
3 - Disgust                 2 - Sad
4 - Happy                   3 - Angry
5 - Sad                     4 - Surprise
6 - Angry                   5 - Fear
7 - Neutral                 6 - Disgust
"""

label_map = {
    "1": 4,  # Surprise
    "2": 5,  # Fear
    "3": 6,  # Disgust
    "4": 1,  # Happy
    "5": 2,  # Sad
    "6": 3,  # Angry
    "7": 0   # Neutral
}

X = []
y = []

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "RAF_DB", "DATASET", "train")


#############################################################


for folder_name in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder_name)

    if folder_name not in label_map:
        print("Unexpected folder - ", folder_name)
        continue

    label = label_map[folder_name]

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print("Image missing from file ", file)
            continue
        else:
            print("Load Image: Success for file ", file)

        #img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        landmark_result = detect_landmarks(img, landmarker)
        if landmark_result is None or not landmark_result.face_landmarks:
            print("    Landmarks not detected in file ", file, " - path=", img_path)
            continue
        else:
            print("Landmarks: Success for file ", file)

        landmarks = landmark_result.face_landmarks[0]
        h, w = img.shape[:2]
        points = [
            (lm.x * w, lm.y * h)
            for lm in landmarks
        ]

        features = extract_features(points)
        if features is None:
            print("    Features not extracted from file ", file, " - path=", img_path)
            continue
        else:
            print("Features: Success for file ", file)

        X.append(features)
        y.append(label)


if len(X) == 0:
    raise RuntimeError("No samples collected")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)


mean = X.mean(axis=0)
std = X.std(axis=0)

X = (X - mean) / (std + 1e-6)


# save dataset
os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

np.save("data/processed/mean.npy", mean)
np.save("data/processed/std.npy", std)

print("Done:")
print("Samples:", len(X))
print("Feature size:", X.shape[1])
print("Mean:", mean)
print("std:", std)
print(np.bincount(y))