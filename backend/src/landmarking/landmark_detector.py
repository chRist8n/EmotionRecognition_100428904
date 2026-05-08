import sys
import os
import mediapipe as mp


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


def detect_landmarks(frame, landmarker):

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = landmarker.detect(mp_image)


    if result.face_landmarks:
        return result
    else:
        return None

    

    

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ensure root is in path
    import backend.src.webcam_debug as webcam_debug  # run webcam_debug.py
    sys.exit()
    
    # img_raw = cv2.imread("data/debugging_images/test_normalized.jpg")

    # # apply some processing
    # img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # landmarks = detect_landmarks(img)

    