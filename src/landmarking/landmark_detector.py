import sys
import os
import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# BaseOptions = python.BaseOptions
# FaceLandmarker = vision.FaceLandmarker
# FaceLandmarkerOptions = vision.FaceLandmarkerOptions
# VisionRunningMode = vision.RunningMode

# model_path = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE,
#     num_faces=1
# )

# landmarker = FaceLandmarker.create_from_options(options)


def build_face_box(frame, points):
    #get all x and y coords and find mesh's edges in each direction
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    #define face width and height
    face_w = x_max - x_min
    face_h = y_max - y_min

    #allow a small margin
    margin_x = int(face_w * 0.25)
    margin_y = int(face_h * 0.35)

    #define a bounding box
    x = x_min - margin_x
    y = y_min - margin_y
    w = face_w + 2 * margin_x
    h = face_h + 2 * margin_y

    #clamp box to image bounds
    h_img, w_img = frame.shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    return (x, y, w, h)

def validate_landmarks(frame, landmarks):
    if landmarks is None or len(landmarks.face_landmarks) == 0:
        return {"valid": False}

    h, w = frame.shape[:2]
    face = landmarks.face_landmarks[0]  # first detected face
    points = [(int(lm.x * w), int(lm.y * h)) for lm in face]

    # analyse points to ensure face mesh is valid and accurate
    valid = True

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    area_ratio = (face_w * face_h) / (w * h)

    LEFT_EYE = 33       # rough indices for MediaPipe FaceMesh
    RIGHT_EYE = 263
    MOUTH = 13
    eye_y = (points[LEFT_EYE][1] + points[RIGHT_EYE][1]) / 2
    mouth_y = points[MOUTH][1]

    if eye_y > mouth_y:
        # invalid (flipped / bad detection)
        valid = False
    if area_ratio < 0.05:
        # too far away
        valid = False
    if area_ratio > 0.6:
        # too close
        valid = False
    if face_w < 30 or face_h < 30:
        # clearly wrong detection
        valid = False

    return {
        "valid": valid,
        "points": points,
        "area_ratio": area_ratio
    }

def detect_landmarks(face_crop, landmarker):

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=face_crop
    )

    result = landmarker.detect(mp_image)


    if result.face_landmarks:
        return result
    else:
        return None

    

    

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import webcam_debug  # run webcam_debug.py
    sys.exit()
    
    # img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # # apply some processing
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # landmarks = detect_landmarks(img)

    