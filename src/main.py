from PIL import Image
import numpy as np

from detection.face_detector import detect_faces
def main():
    #test face detector:

    img = Image.open("data\debugging_images\test_normalized.npy")

    detections = detect_faces(img)
    print("Detections: ", detections)    



if __name__ == "__main__":
   main()