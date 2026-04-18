"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""

import traceback
import preprocess
import detection.face_detector as detector
import landmarking.landmark_detector as landmarker
import cv2
import importlib
import numpy as np


# open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

try:
    frame_count = 0
    while True:
        #reload all imported files from this project every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            importlib.reload(preprocess)
            importlib.reload(detector)
            importlib.reload(landmarker)

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

        #1) Preprocess
        frame = preprocess.preprocess_image(raw_frame)

        #2.1) Detect face
        detection = detector.detect_faces(frame)
        if detection is not None:
            x, y, w, h = detection
        else:
            continue # if nothing is detected, move on to next frame

        #2.2) Crop to new work area (== detection)
        orig_h, orig_w = raw_frame.shape[:2]
        proc_h, proc_w = frame.shape[:2]

        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h

        #3.1) Map detection back to raw_frame
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)
        w_orig = int(w * scale_x)
        h_orig = int(h * scale_y)

        # crop raw frame using scaled coordinates
        face_crop = raw_frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]

        # check crop is valid
        if face_crop.size == 0:
            print("Error: Subject may be too close or out of frame")
            continue    # skip frame if something went wrong

        # ensure size = (256,256)
        face_crop = cv2.resize(face_crop, (256, 256))
        

        #4) Find facial landmarks

        # # normalise face_crop
        # crop_norm = preprocess.preprocess_image(face_crop)

        # test for debugging landmark detection:
        landmarks = landmarker.detect_landmarks(face_crop)

        debug = face_crop.copy()
        if landmarks.face_landmarks:
            for landmark in landmarks.face_landmarks[0]:
                x = int(landmark.x * face_crop.shape[1])
                y = int(landmark.y * face_crop.shape[0])

                cv2.circle(debug, (x, y), 1, (0,255,0), -1)



        ## PIPELINE END ##

        ####################################################################

        ## OUTPUT: ##

        
        # cv2.imshow("Webcam Debug", face_crop)
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
    