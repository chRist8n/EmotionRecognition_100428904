import sys
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


def detect_landmarks(face_crop):

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=face_crop
    )

    result = landmarker.detect(mp_image)

    return result

    

    

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()
    
    # img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # # apply some processing
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # landmarks = detect_landmarks(img)

    