import cv2
import numpy as np

# Load pretrained detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    return img


def detect_face(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    # return largest face
    return max(faces, key=lambda f: f[2] * f[3])


def normalize_face(image: np.ndarray, box, size=256):
    x, y, w, h = box
    face = image[y:y+h, x:x+w]

    face = cv2.resize(face, (size, size))
    face = face / 255.0

    return face


def process_image(path: str):
   img = load_image(path)
   box = detect_face(img)

   if box is None:
       return None

   face = normalize_face(img, box)
   return face


#from face_preprocess import process_image
#import cv2

#face = process_image("test.jpg")

#if face is None:
#    print("No face detected")
#else:
#    print("Face detected:", face.shape)

