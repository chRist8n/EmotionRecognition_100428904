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
#from classification.MLP_test import model as test_model
import classification.MLP as mlp
from landmarking.smoothing import smooth_predictions


## Debugging ##
import time
import psutil, os

process = psutil.Process(os.getpid())
cpu_samples = []

fps_counter = 0
fps_start = time.time()

fps_samples = []
latencies = []
## Debugging end ##


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

#create MLP
X_train = np.load("backend/data/processed/train/X.npy")
y_train = np.load("backend/data/processed/train/y.npy")

model = mlp.MLP(input_size=15, hidden_size_1=64, hidden_size_2=32, output_size=7)
model.train(X_train, y_train, epochs=300)



label_map = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Surprise",
    5: "Fear",
    6: "Disgust"
}


outputStage = 0


HISTORY_SIZE = 5
emotion_history = deque(maxlen=HISTORY_SIZE)
prob_history = deque(maxlen=HISTORY_SIZE)

prev_feedback = None
prev_smooth_landmarks = None
prev_smooth_box = None

mean = np.load("backend/data/processed/train/mean.npy")
std = np.load("backend/data/processed/train/std.npy")


# open webcam (0 = default camera)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Webcam failed to open")

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


        ## Debugging ##
        start_time = time.perf_counter()
        ## Debugging end ##

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
        features = feature_extraction.extract_features(na_points)

        #normalise features

        features = np.array(features)
        features = (features - mean) / (std + 1e-6)
        features = features.reshape(1, -1)

        # 6) classify

        preds = model.predict(features)
        pred = preds[0]
        probs = model.predict_proba(features)[0]
        confidence = max(probs)

        
        smooth_probs = smooth_predictions(prob_history, probs)
        prob_history.append(smooth_probs)

        pred = np.argmax(smooth_probs)
        avg_conf = np.max(smooth_probs)

        # avg_probs = np.mean(prob_history, axis=0)
        # pred = np.argmax(avg_probs)
        # avg_conf = np.max(avg_probs)

        if avg_conf < 0.3:#confidence < 0.15:
            pred_label = "uncertain"
        else:
            pred_label = pred
            #emotion_history.append(pred)
            #pred_label = max(set(emotion_history), key=emotion_history.count)


        ## Debugging ##
        # FPS
        fps_counter += 1

        if time.time() - fps_start >= 1.0:
            print("FPS:", fps_counter)
            fps_samples.append(fps_counter)
            fps_counter = 0
            fps_start = time.time()

        # latency
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

        # CPU usage
        if frame_count % 10 == 0:
            cpu = process.cpu_percent(interval=None)
            cpu_samples.append(cpu)
        ## Debugging end ##

        ## PIPELINE END ##

        ####################################################################

        ## OUTPUT: ##
        

        debug = raw_frame.copy()
                
        if (outputStage > 0):
            debug = aligned_crop.copy()
            h, w = debug.shape[:2]  

        if (outputStage > 1):
            #draw face mesh landmarks
            #points need to be resized from (0-1) to debug.shape

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

        if (outputStage > 2):
            def draw_line(p1, p2, color, label):
                x1, y1 = na_points[p1]
                x2, y2 = na_points[p2]

                x1, y1 = int(x1 * w), int(y1 * h)
                x2, y2 = int(x2 * w), int(y2 * h)

                cv2.line(debug, (x1, y1), (x2, y2), color, 1)

                mx, my = (x1 + x2)//2, (y1 + y2)//2
                cv2.putText(debug, label, (mx, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 0), 1)

            def put_value(text, i):
                cv2.putText(debug, text, (10, 20 + i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)

            features_list = features.flatten() if isinstance(features, np.ndarray) else features

            # ---------------- EYES ---------------- #
            draw_line(159, 145, (0,255,0), "L_eye_open")
            draw_line(386, 374, (0,255,0), "R_eye_open")

            # put_value(f"eye_open_mean: {features_list[0]:.3f}", 0)
            # put_value(f"eye_open_diff: {features_list[1]:.3f}", 1)

            # ---------------- BROWS ---------------- #
            draw_line(33, 70, (255,0,0), "L_brow")
            draw_line(362, 300, (255,0,0), "R_brow")

            # put_value(f"brow_height_mean: {features_list[2]:.3f}", 2)
            # put_value(f"brow_asymmetry: {features_list[3]:.3f}", 3)
            # put_value(f"brow_shape: {features_list[4]:.3f}", 4)

            # ---------------- MOUTH ---------------- #
            draw_line(13, 14, (0,0,255), "mouth_open")
            draw_line(78, 308, (0,0,255), "mouth_width")

            # put_value(f"mouth_open: {features_list[5]:.3f}", 5)
            # put_value(f"mouth_width: {features_list[6]:.3f}", 6)
            # put_value(f"mouth_curve: {features_list[7]:.3f}", 7)
            # put_value(f"mouth_asym: {features_list[8]:.3f}", 8)
            # put_value(f"mouth_drop: {features_list[9]:.3f}", 9)

            # ---------------- GLOBAL ---------------- #
            draw_line(33, 78, (255,255,0), "eye-mouth")

            # put_value(f"eye_mouth_dist: {features_list[10]:.3f}", 10)

            # ---------------- EXPRESSIVE ---------------- #
            # put_value(f"smile_strength: {features_list[11]:.3f}", 11)
            # put_value(f"eye_brow_ratio: {features_list[12]:.3f}", 12)
            # put_value(f"mouth_eye_balance: {features_list[13]:.3f}", 13)
            # put_value(f"face_asymmetry: {features_list[14]:.3f}", 14)

        if (outputStage > 3):
            cv2.putText(
                debug,
                f"{label_map.get(pred_label)} ({avg_conf:.2f})",
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
            ## Debug Outputs ##
            print("Avg FPS:", sum(fps_samples)/len(fps_samples))
            print("Avg latency:", sum(latencies)/len(latencies))
            print("Avg CPU usage:", sum(cpu_samples)/len(cpu_samples))
            ## Debug Outputs end ##
            break
        elif cv2.waitKey(1) & 0xFF == ord('0'):
            print("Outputting Stage 0.")
            outputStage = 0
        elif cv2.waitKey(1) & 0xFF == ord('1'):
            print("Outputting Stage 1.")
            outputStage = 1
        elif cv2.waitKey(1) & 0xFF == ord('2'):
            print("Outputting Stage 2.")
            outputStage = 2
        elif cv2.waitKey(1) & 0xFF == ord('3'):
            print("Outputting Stage 3.")
            outputStage = 3
        elif cv2.waitKey(1) & 0xFF == ord('4'):
            print("Outputting Stage 4.")
            outputStage = 4
except Exception as e:
    #incase of exception, print to console
    print(f"Debugger - Error: {e} \n" )
    traceback.print_exc()
finally:
    #always release and destroy everything on close
    cap.release()
    cv2.destroyAllWindows()
    