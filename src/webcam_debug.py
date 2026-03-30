"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""
import preprocess
import detection.face_detector as detector
import cv2
import importlib


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



        ## PIPELINE START:

        """
        Steps to be completed in pipeline: (ticked with a [X] if implemented)
            -   recieve frame from zoom []
            1)  apply preprocessing [X]
            2)  find face and create bounding box [X]
            3)  crop to bounding box []
            4)  find facial landmarks []
            5)  align and normalise []
            6)  classify []
            -   output []

        In this debugger, the input is simulated with a live webcam feed and
        is outputted (at any given stage) in a cv2.imshow() window.
        """

        #preprocess
        frame = preprocess.preprocess_image(raw_frame)

        #detect face
        detection = detector.detect_faces(frame)
        face_crop = None    # initialise face_crop to avoid potential crashes
        if detection is not None:
            x, y, w, h = detection
            face_crop = frame[y:y+h, x:x+w] # prepare for cropping

        #crop to new work area
        if face_crop is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (256, 256))

        ## PIPELINE END



        ## OUTPUT FULL FRAME
        display_frame = (frame * 255).astype("uint8")

        #draw a rectangle on the image so the ROI is easily visible
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Webcam View", display_frame)

        ##OUTPUT CROPPED VIEW
        if face_crop is not None:
            cropped_frame = (face_crop * 255).astype("uint8")
            cv2.imshow("Cropped View", cropped_frame)


        #exit with [`] key
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break
except Exception as e:
    #incase of actual exception, print to console
    print(f"Debugger - Error: \n, {e}" )
finally:
    #always release and destroy everything on close
    cap.release()
    cv2.destroyAllWindows()
    