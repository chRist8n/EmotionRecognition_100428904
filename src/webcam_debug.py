"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""

import os
import traceback
#import preprocess
#import detection.face_detector as detector
import landmarking.landmark_detector as landmarking
import cv2
import importlib
import numpy as np
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


# open webcam (0 = default camera)
cap = cv2.VideoCapture(1)

try:
    frame_count = 0
    while True:
        #reload all imported files from this project every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            importlib.reload(landmarking)

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
            #fallback/ skip frame
            continue
            #    NOTE: if frequent invalid detects becomes 
            #    a problem, handle here
        else:
            points = feedback["points"]
            area_ratio = feedback["area_ratio"]


        # 3) find bounding box for face

        bounding_box = landmarking.build_face_box(raw_frame, points)
        x, y, w, h = bounding_box
        face_crop = raw_frame[y:y+h, x:x+w]

        # 4) align + normalise
        


        ## PIPELINE END ##

        ####################################################################

        ## OUTPUT: ##

        #draw face mesh landmarks
        debug = raw_frame.copy()
        if landmarks.face_landmarks:
            for landmark in landmarks.face_landmarks[0]:
                x = int(landmark.x * debug.shape[1])
                y = int(landmark.y * debug.shape[0])

                cv2.circle(debug, (x, y), 1, (0,255,0), -1)

        # #highlight bounding box
        # x, y, w, h = bounding_box
        # cv2.rectangle(debug, (x, y), (x + w, y + h), (0,0,255), 1)

        cv2.imshow("Webcam Debug", debug)



        # # initialise full frame for display and highlight ROI
        # display_frame = (frame * 255).astype("uint8")
        # cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # # initialise cropped view for display
        # display_crop = (face_crop * 255).astype("uint8")

        # # combine and output
        # combined = cv2.hconcat([display_frame, display_crop])
        # cv2.imshow("Webcam Debug", combined)
        

        ## OUTPUT END ##

        ####################################################################

        ## EXIT CONDITIONS:

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
    