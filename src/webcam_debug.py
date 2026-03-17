"""
This file runs a loop to display a live webcam feed which 
has been fedthrough the algorithm(s) for debugging purposes.
-   Quit with [`] key
"""
from preprocess import preprocess_image
from detection.face_detector import detect_faces
import cv2


# open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:         # ret will be true if frame captured successfully
        break

    #preprocess

    #detect face(s)

    #display frame
    cv2.imshow("Webcam", frame)

    #exit with [`] key
    if cv2.waitKey(1) & 0xFF == ord('`'):
        break

cap.release()
cv2.destroyAllWindows()