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

        #preprocess
        frame = preprocess.preprocess_image(raw_frame)

        #detect face(s)
        detections = detector.detect_faces(frame)

        #display the frame
        display_frame = (frame * 255).astype("uint8")

        for (x, y, w, h) in detections:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Webcam", display_frame)

        #exit with [`] key
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break
except Exception as e:
    #incase of actual exception, print to console
    print(f"Error: \n, {e}" )
finally:
    #always release and destroy everything on close
    cap.release()
    cv2.destroyAllWindows()
    