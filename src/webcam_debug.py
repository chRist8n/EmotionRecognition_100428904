"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Uses importlib.reload() to update file changes while
    debugging
-   Quit with [`] key
"""
from preprocess import preprocess_image
from detection.face_detector import detect_faces
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
            importlib.reload(preprocess_image)
            importlib.reload(detect_faces)
            frame_count = 0

        #capture frame
        ret, frame = cap.read() 
        if not ret:         # ret will be true if frame captured successfully
            break

        #preprocess
        frame = preprocess_image()

        #detect face(s)

        #display frame
        cv2.imshow("Webcam", frame)

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
    