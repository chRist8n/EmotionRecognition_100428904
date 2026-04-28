"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""

import os
import traceback
import cv2
import importlib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

#import preprocess
#import detection.face_detector as detector
import landmarking.landmark_detector as landmarking
import landmarking.cropping as cropping
import landmarking.smoothing as smoothing
import landmarking.feature_extraction as feature_extraction
from classification.MLP_test import model as test_model



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


label_map = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Surprise",
    5: "Fear",
    6: "Disgust"
}


emotion_history = deque(maxlen=5)

prev_feedback = None
prev_smooth_landmarks = None
prev_smooth_box = None


# open webcam (0 = default camera)
cap = cv2.VideoCapture(1)

try:
    frame_count = 0
    while True:
        #reload all imported files from this project every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            importlib.reload(landmarking)
            importlib.reload(cropping)
            importlib.reload(smoothing)
            importlib.reload(feature_extraction)

        #capture frame
        ret, raw_frame = cap.read() 
        if not ret:         # ret will be true if frame captured successfully
            continue        # if not ret, skip this frame



        ####################################################################

        ## PIPELINE START: ##

        """
        Steps to be completed in pipeline: (ticked with a [X] if implemented)
            -   recieve frame from zoom []
            1)  detect face + create face mesh []
            2)  validation + quality checks []
                - fallback to previous frame if needed
                - apply smoothing
            3)  find bounding box for face []
                - crop raw frame to new ROI
            4)  align and normalise []
            5)  feature extraction []
            6)  classify []
            -   output []

        In this debugger, the input is simulated with a live webcam feed and
        is outputted (at any given stage) in a cv2.imshow() window.
        """



        # 1) detect face + create face mesh

        landmarks = landmarking.detect_landmarks(raw_frame, landmarker)


        # 2) validation + quality checks

        feedback = landmarking.validate_landmarks(raw_frame, landmarks)
        if not feedback["valid"]:
            if prev_feedback is not None:
                #use previous valid landmarks
                feedback = prev_feedback
            else:
                #fallback/ skip frame
                continue
        points = feedback["points"]
        area_ratio = feedback["area_ratio"]
        prev_feedback = feedback

        # #apply smoothing
        # points = smoothing.smooth_landmarks(prev_smooth_landmarks, points)
        # prev_smooth_landmarks = points


        # 3) find bounding box for face

        bounding_box = cropping.build_face_box(raw_frame, points)

        #smooth ROI box
        smooth_box = smoothing.smooth_box(prev_smooth_box, bounding_box)
        prev_smooth_box = smooth_box

        #map all to int
        bx, by, bw, bh = map(int, smooth_box)

        #crop image to face
        face_crop = raw_frame[by:by+bh, bx:bx+bw]


        # 4) align + normalise

        #match landmarks to new size/aspect ratio
        points_cropped = [(px - bx, py - by) for (px, py) in points]


        # 4.1) align
        #define left/right eye centre
        LEFT_EYE_CENTRE = cropping.compute_centre(points_cropped, [33, 133, 160, 159])
        lcx = LEFT_EYE_CENTRE[0]
        lcy = LEFT_EYE_CENTRE[1]
        RIGHT_EYE_CENTRE = cropping.compute_centre(points_cropped, [362, 263, 387, 386])
        rcx = RIGHT_EYE_CENTRE[0]
        rcy = RIGHT_EYE_CENTRE[1]

        #find angle between eyes
        dx = rcx - lcx
        dy = rcy - lcy
        eye_tilt = np.degrees(np.arctan2(dy, dx))

        #rotate image
        ROI_centre = (bw // 2, bh // 2)
        M = cv2.getRotationMatrix2D(ROI_centre, eye_tilt, 1.0)
        aligned_crop = cv2.warpAffine(face_crop, M, (bw, bh))

        #rotate landmarks
        aligned_points = []
        for (x, y) in points_cropped:
            vec = np.array([x, y, 1.0])
            rot_x, rot_y = M @ vec
            aligned_points.append((rot_x, rot_y))

        # 4.2) normalise
        #normalise aligned points
        norm_aligned_points = [(x / bw, y / bh) for (x, y) in aligned_points]

        #clamp norm_aligned
        na_points = [
            (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
            for (x, y) in norm_aligned_points
        ]


        # 5) feature extraction

        # # key features
        # LEFT_EYE = [33, 133, 159, 145]
        # RIGHT_EYE = [362, 263, 386, 374]
        # MOUTH = [13, 14, 78, 308]
        # LEFT_BROW = [70, 63]
        # RIGHT_BROW = [300, 293]

        features = feature_extraction.extract_features(na_points)

        #normalise features
        mean = np.load("data/processed/mean.npy")
        std = np.load("data/processed/std.npy")

        features = np.array(features)
        features = (features - mean) / (std + 1e-6)
        features = features.reshape(1, -1)

        # 6) classify

        #features = np.array(features).reshape(1, -1)
        pred = test_model.predict(features)[0]
        probs = test_model.predict_proba(features)[0]
        confidence = max(probs)

        if confidence < 0.3:
            pred_label = "uncertain"
        else:
            emotion_history.append(pred)
            pred_label = max(set(emotion_history), key=emotion_history.count)



        ## PIPELINE END ##

        ####################################################################

        ## OUTPUT: ##

        #draw face mesh landmarks
        debug = aligned_crop.copy()
        h, w = debug.shape[:2]      #points need to be resized from (0-1) to debug.shape

        if na_points:
            for (x, y) in na_points:
                cv2.circle(debug, (int(x * w), int(y * h)), 2, (0,255,0), -1)

        #draw eye centre points
        LEFT_EYE_CENTRE = cropping.compute_centre(na_points, [33, 133, 160, 159])
        RIGHT_EYE_CENTRE = cropping.compute_centre(na_points, [362, 263, 387, 386])

        le_x = int(LEFT_EYE_CENTRE[0] * w)
        le_y = int(LEFT_EYE_CENTRE[1] * h)
        re_x = int(RIGHT_EYE_CENTRE[0] * w)
        re_y = int(RIGHT_EYE_CENTRE[1] * h)

        cv2.circle(debug, (le_x, le_y), 4, (0,0,255), -1)
        cv2.circle(debug, (re_x, re_y), 4, (0,0,255), -1)

        # #draw line between eye centres
        # cv2.line(debug, (le_x, le_y), (re_x, re_y), (0, 255, 255), 1)

        #print features to screen
        # y0 = 20

        # for i, f in enumerate(features):
        #     text = f"F{i}: {f:.3f}"
        #     cv2.putText(
        #         debug,
        #         text,
        #         (10, y0 + i * 20),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 0, 0),
        #         1,
        #         cv2.LINE_AA
        #     )

        cv2.putText(
            debug,
            f"{label_map.get(pred_label)} ({confidence:.2f})",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.imshow("Webcam Debug", debug)

        ## OUTPUT END ##

        ####################################################################

        ## EXIT CONDITIONS + ERROR HANDLING:

        #exit with [`] key
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break
except Exception as e:
    #incase of exception, print to console
    print(f"Debugger - Error: {e} \n" )
    traceback.print_exc()
finally:
    #always release and destroy everything on close
    cap.release()
    cv2.destroyAllWindows()
    