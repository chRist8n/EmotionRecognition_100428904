"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""
import sys
import preprocess
import detection.face_detector as detector
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

        #capture frame
        ret, raw_frame = cap.read() 
        if not ret:         # ret will be true if frame captured successfully
            continue        # if not ret, skip this frame



        ####################################################################

        ## PIPELINE START: ##

        """
        Steps to be completed in pipeline: (ticked with a [X] if implemented)
            -   recieve frame from zoom []
            1)  apply preprocessing [X]
            2)  find face and create bounding box [X]
            3)  crop raw frame to bounding box []
                - reapply preprocessing to cropped version of raw frame
            4)  find facial landmarks []
            5)  align and normalise []
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

        #2.2) Crop to new work area (= detection)
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

        #3.2) Apply secondary cropping for tighter fit to face

        # YCrCb skin-tone mask
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        skin_mask = (
            (Cr >= 133) & (Cr <= 173) &
            (Cb >= 77)  & (Cb <= 127)
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin_mask, connectivity=8)

        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            skin_mask = (labels == largest_label).astype(np.uint8)

        skin_coords = np.column_stack(np.where(skin_mask > 0))

        if len(skin_coords) > 0:
            top = np.min(skin_coords[:, 0])
            bottom = np.max(skin_coords[:, 0])
            left = np.min(skin_coords[:, 1])
            right = np.max(skin_coords[:, 1])
        else:
            # fallback - keep original crop
            top = 0
            bottom = face_crop.shape[0]
            left = 0
            right = face_crop.shape[1]

        # estimated face region size
        face_h = bottom - top
        face_w = right - left

        # add padding
        pad_y = int(face_h * 0.25)
        pad_x = int(face_w * 0.20)

        top = max(0, top - pad_y)
        bottom = min(face_crop.shape[0], bottom + pad_y)
        left = max(0, left - pad_x)
        right = min(face_crop.shape[1], right + pad_x)

        # tighter crop inside original face_crop
        refined_crop = raw_frame[y_orig + top:y_orig + bottom, x_orig + left:x_orig + right]

        # if refined_crop.size > 0:
        #     face_crop = refined_crop
        # else:
        #     continue    # skip frame if something went wrong

        #face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        





        ## PIPELINE END ##



        ####################################################################

        ## OUTPUT: ##


        cv2.imshow("Webcam Debug", face_crop)



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
    lineno = sys.exc_info()[-1].tb_lineno
    print(f"Debugger - Error: \n {e} at line {lineno}" )
finally:
    #always release and destroy everything on close
    cap.release()
    cv2.destroyAllWindows()
    